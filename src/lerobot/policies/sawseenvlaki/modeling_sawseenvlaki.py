# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SawSeenVLA + Knowledge Insulation (KI) with FAST action tokens.

Same flow-matching action expert as SawSeenVLA, plus:

1. ``fast_embed`` (Embedding(fast_vocab_size, d_vlm)) — dedicated
   input projection for FAST tokens, fully trainable via
   ``modules_to_save``. The output projection reuses the VLM's
   existing ``lm_head`` (sliced over the last ``fast_vocab_size``
   rows) with LoRA adapters, so the FAST CE flows through
   pretrained text-vocabulary geometry that the LoRA deforms —
   closer in spirit to the π0.5 / π0.6 recipe than a fully-random
   output head. SmolVLM2 has untied 47 M-param embed/head matrices,
   so naive vocab extension via ``modules_to_save`` would add ~95 M
   trainable params and OOM 24 GB cards; the dedicated
   ``fast_embed`` + sliced ``lm_head`` path costs only ~2 M (embed)
   + ~0.8 M (lm_head LoRA at r=16).

2. FAST tokens enter the prefix between language and state with
   ``att_mask=[1, 1, ..., 1]`` so each FAST token is its own block —
   causal within the FAST suffix. The action expert (suffix) is
   explicitly masked away from FAST keys so it can't read its own
   answer.

3. A ``.detach()`` boundary on the VLM K/V tensors going into the
   action expert (via ``detach_kv_for_expert=True`` on the wrapper
   forward) keeps the flow-matching gradient from updating the
   VLM — only the FAST CE updates VLM weights, through LoRA
   adapters on text_model q/v and on lm_head. With KI off the
   policy is structurally equivalent to SawSeenVLA.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from ..rtc.modeling_rtc import RTCProcessor
from ..sawseenvla.modeling_sawseenvla import (
    SawSeenVLAPolicy,
    VLAFlowMatching,
    make_att_2d_masks,
    pad_tensor,
)
from .configuration_sawseenvlaki import SawSeenVLAKIConfig


class VLAFlowMatchingKI(VLAFlowMatching):
    """SawSeenVLA flow-matching expert + dedicated FAST head/embed.

    See :class:`SawSeenVLAKIPolicy` for the architecture overview.
    """

    def __init__(self, config: SawSeenVLAKIConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.config = config  # type: SawSeenVLAKIConfig
        d_vlm = self.vlm_with_expert.config.text_config.hidden_size
        # Dedicated FAST INPUT projection. ``fast_embed(token_id)``
        # produces an embedding in the VLM hidden space that is
        # consumed by the transformer like any language token. The
        # corresponding OUTPUT projection reuses the VLM's existing
        # ``lm_head`` — see ``forward_ki`` — so the lm_head's
        # LoRA-adapted rows over the last ``fast_vocab_size`` indices
        # serve as the next-FAST-token classifier.
        self.fast_embed = nn.Embedding(self.config.fast_vocab_size, d_vlm)
        # Index of the first row of ``vlm.lm_head`` that we treat as
        # FAST token ID 0. Tail of the SmolLM2 vocab — token IDs there
        # are real (rare) BPE pieces; with LoRA those rows can deform
        # to align with FAST-token directions.
        vocab_size = int(self.vlm_with_expert.vlm.config.text_config.vocab_size)
        self._fast_id_offset = vocab_size - self.config.fast_vocab_size

    # ------------------------------------------------------------------
    # Prefix construction with FAST tokens spliced between lang & state.
    # ------------------------------------------------------------------
    def embed_prefix_ki(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: torch.Tensor,
        fast_tokens: torch.Tensor,
        fast_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Mirror of :meth:`VLAFlowMatching.embed_prefix` with FAST splice.

        Layout:

            [images …]      att=0  (block 0, bidirectional within)
            [lang …]        att=0  (still block 0)
            [FAST_0 … N-1]  att=1  each — own block, causal AR within
            [state]         att=1  (block N+1)

        Returns the standard ``(embs, pad_masks, att_masks)`` triple
        plus the (start, end) prefix-position range covered by the FAST
        tokens — the caller uses it to mask the action expert away from
        FAST and to slice FAST hidden states for the CE head.
        """
        embs = []
        pad_masks = []
        att_masks: list[int] = []

        # ---- images ----
        for img, img_mask in zip(images, img_masks, strict=False):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * image_start_mask.shape[-1]
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask_b = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask_b)
            att_masks += [0] * num_img_embs

            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * image_end_mask.shape[1]

        # ---- language ----
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # ---- FAST tokens (each own block, att=1 → causal within FAST) ----
        # fast_tokens: (B, N_fast). fast_embed produces (B, N_fast, d_vlm).
        # Pad-fix: out-of-range / pad IDs get clamped to 0; the loss
        # mask zeroes them out anyway, but the embedding lookup must
        # never see negative or >= vocab IDs.
        fast_tokens_safe = fast_tokens.clamp(min=0, max=self.config.fast_vocab_size - 1)
        fast_emb = self.fast_embed(fast_tokens_safe)
        fast_emb_dim = fast_emb.shape[-1]
        fast_emb = fast_emb * math.sqrt(fast_emb_dim)
        fast_start = sum(t.shape[1] for t in embs)  # in prefix coords
        embs.append(fast_emb)
        pad_masks.append(fast_masks.bool())
        n_fast = fast_emb.shape[1]
        att_masks += [1] * n_fast
        fast_end = fast_start + n_fast

        # ---- state ----
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device
        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1] * states_seq_len

        # ---- finalize ----
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks_t = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks_t = pad_tensor(att_masks_t, self.prefix_length, pad_value=0)

        att_masks_t = att_masks_t.expand(bsize, -1)
        return embs, pad_masks, att_masks_t, fast_start, fast_end

    # ------------------------------------------------------------------
    # KI training forward — returns (action_losses, ce_loss).
    # ------------------------------------------------------------------
    def forward_ki(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        fast_tokens: torch.Tensor,
        fast_masks: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """KI-mode training step. Returns ``(action_losses, ce_loss)``.

        The action loss is the standard flow-matching MSE (per-action
        residuals). The CE loss is next-FAST-token cross-entropy on the
        VLM's hidden states at FAST positions, masked by ``fast_masks``
        and shifted (``logit_i`` predicts ``token_{i+1}``).
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, fast_start, fast_end = self.embed_prefix_ki(
            images, img_masks, lang_tokens, lang_masks, state, fast_tokens, fast_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        prefix_len = prefix_embs.shape[1]
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        # The action expert (suffix) cross-attends to VLM K/V at all
        # prefix positions. Block its attention to FAST keys so it
        # cannot read its own answer. Without this, training would
        # collapse to "look at FAST, copy to suffix".
        att_2d_masks[:, prefix_len:, fast_start:fast_end] = False

        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (prefix_out, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            detach_kv_for_expert=True,
        )

        # ---- action loss (flow-matching MSE) ----
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        action_losses = F.mse_loss(u_t, v_t, reduction="none")

        # ---- CE loss on FAST positions ----
        # Hidden states at FAST positions: prefix_out[:, fast_start:fast_end].
        # Project via the VLM's lm_head, then slice to the FAST range
        # before softmax. ``fast_token_ids`` arrive in [0, fast_vocab_size);
        # the matching logits live at lm_head rows
        # [_fast_id_offset, _fast_id_offset + fast_vocab_size). Slicing
        # before CE means we don't pay the softmax cost over ~49k IDs
        # nor backprop a "suppress non-FAST logits" gradient at FAST
        # positions (those positions never produce text tokens — the
        # FAST and text vocabularies are disjoint by convention).
        fast_hidden = prefix_out[:, fast_start:fast_end].to(dtype=torch.float32)
        lm_head = self.vlm_with_expert.vlm.lm_head
        full_logits = lm_head(fast_hidden)
        fast_logits = full_logits[
            ..., self._fast_id_offset : self._fast_id_offset + self.config.fast_vocab_size
        ]
        # Shifted targets: logits[:, :-1] predict tokens[:, 1:].
        # Loss mask: we count a position only if BOTH the input token
        # at that position AND the target (next) token were real
        # (not padding).
        labels = fast_tokens[:, 1:]
        target_valid = fast_masks[:, 1:].bool()
        input_valid = fast_masks[:, :-1].bool()
        valid = target_valid & input_valid
        ce = F.cross_entropy(
            fast_logits[:, :-1].reshape(-1, self.config.fast_vocab_size),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        ce = ce * valid.to(ce.dtype)
        denom = valid.sum().clamp_min(1).to(ce.dtype)
        ce_loss = ce.sum() / denom

        return action_losses, ce_loss


class SawSeenVLAKIPolicy(SawSeenVLAPolicy):
    """SawSeenVLA + KI training (FAST CE head + insulated VLM gradient).

    With ``ki_enabled=False`` (default at instantiation, no FAST tokens
    in the batch), behavior is identical to :class:`SawSeenVLAPolicy`.
    """

    config_class = SawSeenVLAKIConfig
    name = "sawseenvlaki"

    def __init__(self, config: SawSeenVLAKIConfig, **kwargs):
        # Initialize PreTrainedPolicy (skip SawSeenVLAPolicy's
        # ``self.model = VLAFlowMatching(...)`` — we want VLAFlowMatchingKI).
        super(SawSeenVLAPolicy, self).__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatchingKI(config, rtc_processor=self.rtc_processor)
        self.reset()

    # ------------------------------------------------------------------
    # Training forward.
    # ------------------------------------------------------------------
    def forward(
        self,
        batch: dict[str, Tensor],
        noise=None,
        time=None,
        reduction: str = "mean",
    ) -> dict[str, Tensor]:
        # KI off → SawSeenVLA-equivalent path.
        if not self.config.ki_enabled:
            return super().forward(batch, noise=noise, time=time, reduction=reduction)

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        fast_tokens = batch.get(ACTION_TOKENS)
        fast_masks = batch.get(ACTION_TOKEN_MASK)
        if fast_tokens is None or fast_masks is None:
            raise ValueError(
                "ki_enabled=True but the batch is missing FAST action tokens. "
                "Ensure the preprocessor pipeline includes "
                "SawSeenVLAKIFastActionTokenizerProcessorStep (it is added "
                "automatically by make_sawseenvlaki_pre_post_processors when "
                "ki_enabled=True)."
            )

        action_losses, ce_loss = self.model.forward_ki(
            images, img_masks, lang_tokens, lang_masks, state, actions,
            fast_tokens=fast_tokens, fast_masks=fast_masks,
            noise=noise, time=time,
        )

        original_action_dim = self.config.action_feature.shape[0]
        action_losses = action_losses[:, :, :original_action_dim]
        loss_dict: dict[str, float | Tensor] = {
            "loss_action_after_forward": action_losses.clone().mean().item(),
            "loss_ki_ce": ce_loss.detach().item(),
        }

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            action_losses = action_losses * in_episode_bound.unsqueeze(-1)

        action_losses = action_losses[:, :, : self.config.max_action_dim]

        if reduction == "none":
            if actions_is_pad is None:
                per_sample_loss = action_losses.mean(dim=(1, 2))
            else:
                num_valid = ((~actions_is_pad).sum(dim=1) * action_losses.shape[-1]).clamp_min(1)
                per_sample_loss = action_losses.sum(dim=(1, 2)) / num_valid
            # Combine: per-sample action + scalar CE. CE is broadcast.
            combined = per_sample_loss + self.config.ki_loss_weight * ce_loss
            loss_dict["loss_action"] = per_sample_loss.mean().item()
            loss_dict["loss"] = combined.mean().item()
            return combined, loss_dict

        if actions_is_pad is None:
            action_loss = action_losses.mean()
        else:
            num_valid = ((~actions_is_pad).sum() * action_losses.shape[-1]).clamp_min(1)
            action_loss = action_losses.sum() / num_valid

        total = action_loss + self.config.ki_loss_weight * ce_loss
        loss_dict["loss_action"] = action_loss.detach().item()
        loss_dict["loss"] = total.detach().item()
        return total, loss_dict

    # ------------------------------------------------------------------
    # PEFT targets: SawSeenVLA's text_model q/v regex + ``lm_head`` (so
    # the FAST CE flows through LoRA-adapted output rows) + the
    # ``fast_embed`` module in ``modules_to_save`` (it has no
    # pretrained init to LoRA against).
    # ------------------------------------------------------------------
    def _get_default_peft_targets(self) -> dict[str, any]:
        targets = super()._get_default_peft_targets()
        if self.config.ki_enabled:
            # Extend target_modules to also LoRA-wrap ``lm_head``.
            # ~0.8 M LoRA params at r=16 (Linear(960, 49280) → A(960,r)
            # + B(r, 49280)). Compared to dropping a fully-trainable
            # 2 M ``fast_head`` Linear, this is a net −1.2 M trainable.
            targets["target_modules"] = (
                r"model\.vlm_with_expert\.vlm\."
                r"(model\.text_model\..*\.self_attn\.(q|v)_proj|lm_head)"
            )
        modules_to_save = list(targets.get("modules_to_save", []))
        if "fast_embed" not in modules_to_save:
            modules_to_save.append("fast_embed")
        targets["modules_to_save"] = modules_to_save
        return targets
