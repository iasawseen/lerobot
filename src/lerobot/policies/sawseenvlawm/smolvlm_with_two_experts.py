# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Two-experts variant of SmolVLMWithExpertModel for the Latent Goal
Expert — the implementation of the "Future Sight" expert from
``design/future-sight-implicit-wm.md``.

The shared SmolVLM backbone runs once per forward pass; two flow-matching
experts (action expert + Latent Goal Expert) live next to it as parallel
transformer stacks, layer-interleaved with the VLM exactly like the existing
single-expert wrapper. Each expert has its own weights, depth, width, and
projections — they share *only* the VLM.

The dispatch loop is identical to the parent: at each VLM layer either a
self-attention path (Q/K/V from VLM and both experts concatenated along the
sequence dim) or a cross-attention path (each expert's Q reads VLM K/V via
its own re-projection layers) runs. The post-attention output slicing
generalizes naturally to N streams.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from lerobot.policies.smolvla.smolvlm_with_expert import (
    AutoModel,
    SmolVLMWithExpertModel,
    apply_rope,
    get_intermediate_size,
)


class SmolVLMWithTwoExpertsModel(SmolVLMWithExpertModel):
    """SmolVLM + two parallel flow-matching experts (action + Latent Goal Expert).

    Streams during forward (when both experts are fed):
        inputs_embeds = [prefix_embs, action_suffix_embs, latent_goal_suffix_embs]
        model_layers  = [vlm_layers,  action_expert_layers, latent_goal_expert_layers]

    Either Latent Goal Expert embeddings can be ``None`` to skip that expert at inference
    (e.g. when only running the action expert in Mode 1).
    """

    def __init__(
        self,
        *args,
        latent_goal_expert_width_multiplier: float = 0.75,
        latent_goal_num_expert_layers: int = -1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Build the second (Latent Goal Expert) expert from the original VLM text
        # config, scaled by ``latent_goal_expert_width_multiplier`` — same recipe the
        # parent uses for ``lm_expert`` (action), but with its own width and
        # depth knobs.
        original_hidden = self.vlm.config.text_config.hidden_size
        latent_goal_config = copy.deepcopy(self.vlm.config.text_config)
        latent_goal_hidden = int(original_hidden * latent_goal_expert_width_multiplier)
        latent_goal_config.hidden_size = latent_goal_hidden
        latent_goal_config.intermediate_size = get_intermediate_size(latent_goal_hidden)
        latent_goal_config.num_hidden_layers = self.num_vlm_layers
        if latent_goal_num_expert_layers > 0:
            assert self.num_vlm_layers % latent_goal_num_expert_layers == 0, (
                f"num_vlm_layers ({self.num_vlm_layers}) must be a multiple of "
                f"latent_goal_num_expert_layers ({latent_goal_num_expert_layers})"
            )
            latent_goal_config.num_hidden_layers = latent_goal_num_expert_layers
        self.latent_goal_expert = AutoModel.from_config(latent_goal_config)
        self.latent_goal_expert_hidden_size = latent_goal_config.hidden_size
        self.num_latent_goal_expert_layers = len(self.latent_goal_expert.layers)

        # Cross-attention re-projection: in cross-attn layers each expert's
        # k_proj/v_proj must accept VLM-hidden-sized inputs (since they read
        # VLM K/V) and emit its own hidden-sized outputs. Same trick the
        # parent applies to ``lm_expert``.
        if "cross" in self.attention_mode:
            vlm_text = self.vlm.config.text_config
            for layer_idx in range(len(self.latent_goal_expert.layers)):
                if (
                    self.self_attn_every_n_layers > 0
                    and layer_idx % self.self_attn_every_n_layers == 0
                ):
                    continue
                self.latent_goal_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                    vlm_text.num_key_value_heads * vlm_text.head_dim,
                    latent_goal_config.num_key_value_heads * latent_goal_config.head_dim,
                    bias=latent_goal_config.attention_bias,
                )
                self.latent_goal_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                    vlm_text.num_key_value_heads * vlm_text.head_dim,
                    latent_goal_config.num_key_value_heads * latent_goal_config.head_dim,
                    bias=latent_goal_config.attention_bias,
                )

        self.latent_goal_expert.embed_tokens = None

        # Re-apply the requires_grad policy so the new expert's parameters
        # are configured the same way as ``lm_expert``.
        self._set_latent_goal_expert_requires_grad()

    def _set_latent_goal_expert_requires_grad(self):
        # Mirror the parent's behavior: lm_head is frozen on the expert
        # because we don't generate language tokens from it.
        for name, params in self.latent_goal_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False

    def set_requires_grad(self):
        super().set_requires_grad()
        # ``latent_goal_expert`` may not exist yet on the very first call from the
        # parent's __init__ (it runs before our subclass init builds it).
        if hasattr(self, "latent_goal_expert"):
            self._set_latent_goal_expert_requires_grad()

    # ------------------------------------------------------------------
    # Layer dispatch
    # ------------------------------------------------------------------

    def get_model_layers(self, models: list) -> list:
        """Build per-layer-index lists for VLM + each expert.

        Generalizes the parent's two-stream version to N streams. Each
        expert stream uses its own ``num_expert_layers`` so depth ratios
        relative to VLM are independent per expert.
        """
        per_stream_layers: list[list] = [[] for _ in models]
        # First model is always the VLM; experts follow.
        for i, model in enumerate(models):
            if i == 0:
                # VLM uses one layer per VLM index.
                for layer_idx in range(self.num_vlm_layers):
                    per_stream_layers[i].append(model.layers[layer_idx])
                continue
            num_expert_layers = len(model.layers)
            multiple_of = self.num_vlm_layers // num_expert_layers
            for layer_idx in range(self.num_vlm_layers):
                if multiple_of > 0 and layer_idx > 0 and layer_idx % multiple_of != 0:
                    per_stream_layers[i].append(None)
                else:
                    expert_layer_index = (
                        layer_idx // multiple_of if multiple_of > 0 else layer_idx
                    )
                    per_stream_layers[i].append(model.layers[expert_layer_index])
        return per_stream_layers

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        # When both experts feed embeddings, ``inputs_embeds`` has length 3.
        # When only the action expert fires (e.g. inference Mode 1), length
        # 2 is allowed and we fall back to the parent path for compatibility.
        if inputs_embeds is None or len(inputs_embeds) <= 2:
            return super().forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
            )

        models = [self.get_vlm_model().text_model, self.lm_expert, self.latent_goal_expert]
        model_layers = self.get_model_layers(models)
        for hidden_states in inputs_embeds:
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        num_layers = self.num_vlm_layers
        head_dim = self.vlm.config.text_config.head_dim
        for layer_idx in range(num_layers):
            if (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            ):
                # Self-attn path: parent already iterates ``inputs_embeds`` as
                # a list and concatenates Q/K/V along the sequence axis, so it
                # works for N streams unchanged.
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = self._forward_cross_attn_layer_n(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )

            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = (
                    att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                )
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    att_out = att_output[:, start:end]
                    out_emb = layer.self_attn.o_proj(att_out)

                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values

    def _forward_cross_attn_layer_n(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ):
        """Cross-attn path generalized to N experts (parent only handles 1).

        At each cross-attn layer:
          1. The VLM either does its own self-attention on the prefix tokens
             (when ``past_key_values`` has not been built yet) or reads its
             K/V from the cache.
          2. Each expert reads the VLM's K/V via its own ``k_proj``/``v_proj``
             re-projection layers, with its own Q from its own tokens.

        Each expert slices ``position_ids`` and ``attention_mask`` using its
        own offset within the suffix — the parent's ``attention_mask[:, -N:]``
        shortcut works only for one expert. The returned ``att_outputs`` list
        is index-aligned with ``inputs_embeds`` so the post-attention slicing
        in ``forward`` works unchanged.
        """
        attention_interface = self.get_attention_interface()
        att_outputs: list[torch.Tensor | None] = [None] * len(inputs_embeds)

        prefix_embeds = inputs_embeds[0]
        prefix_only_first_pass = prefix_embeds is not None and not past_key_values

        if prefix_only_first_pass:
            prefix_len = prefix_embeds.shape[1]
            position_id_prefix = position_ids[:, :prefix_len]
            position_id_suffix = position_ids[:, prefix_len:]
            prefix_attention_mask = attention_mask[:, :prefix_len, :prefix_len]

            vlm_layer = model_layers[0][layer_idx]
            hidden_states = vlm_layer.input_layernorm(prefix_embeds)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, vlm_layer.self_attn.head_dim)
            hidden_states = hidden_states.to(dtype=vlm_layer.self_attn.q_proj.weight.dtype)
            vlm_q = vlm_layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            vlm_k = vlm_layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            vlm_v = vlm_layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            vlm_q = apply_rope(vlm_q, position_id_prefix)
            vlm_k = apply_rope(vlm_k, position_id_prefix)

            vlm_att = attention_interface(
                prefix_attention_mask, batch_size, head_dim, vlm_q, vlm_k, vlm_v
            )
            att_outputs[0] = vlm_att

            key_states = vlm_k
            value_states = vlm_v
        else:
            # Cached path: ``position_ids`` is the suffix-only run (length =
            # total suffix len), and ``attention_mask`` has shape
            # (B, total_suffix_len, prefix_len + total_suffix_len).
            position_id_suffix = position_ids
            key_states = None
            value_states = None

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]

        # Each expert reads VLM K/V via its own re-projection layers, sliced
        # by its position within the suffix.
        suffix_offset = 0
        for expert_idx in range(1, len(inputs_embeds)):
            expert_embeds = inputs_embeds[expert_idx]
            if expert_embeds is None:
                continue
            expert_layer = model_layers[expert_idx][layer_idx]
            if expert_layer is None:
                # No layer at this VLM index for this expert (lower depth);
                # skip but still advance ``suffix_offset`` since this stream
                # contributes its tokens to the global sequence layout.
                suffix_offset += expert_embeds.shape[1]
                continue
            expert_len = expert_embeds.shape[1]

            expert_pos = position_id_suffix[:, suffix_offset : suffix_offset + expert_len]
            expert_pos = expert_pos - torch.min(expert_pos, dim=1, keepdim=True).values

            if prefix_only_first_pass:
                row_start = prefix_len + suffix_offset
            else:
                row_start = suffix_offset
            row_end = row_start + expert_len
            expert_attention_mask = attention_mask[
                :, row_start:row_end, : key_states.shape[1]
            ]

            expert_hidden_states = expert_layer.input_layernorm(expert_embeds)
            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)
            expert_hidden_states = expert_hidden_states.to(
                dtype=expert_layer.self_attn.q_proj.weight.dtype
            )
            expert_q = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

            _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(
                *key_states.shape[:2], -1
            )
            expert_k = expert_layer.self_attn.k_proj(_key_states).view(
                *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )

            _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(
                *value_states.shape[:2], -1
            )
            expert_v = expert_layer.self_attn.v_proj(_value_states).view(
                *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )

            expert_q_rope = apply_rope(expert_q, expert_pos)

            expert_att = attention_interface(
                expert_attention_mask,
                batch_size,
                head_dim,
                expert_q_rope,
                expert_k,
                expert_v,
            )
            att_outputs[expert_idx] = expert_att
            suffix_offset += expert_len

        return att_outputs, past_key_values
