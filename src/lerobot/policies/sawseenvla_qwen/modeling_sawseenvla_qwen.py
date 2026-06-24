"""SawSeenVLA-Qwen modeling.

Architecture (Scheme B):
  - Qwen3.5-0.8B as a black-box VL encoder. Reads per-layer hidden states from the
    6 full-attention anchor layers (configurable via `qwen_cross_attn_layer_indices`).
  - Independent ActionExpertDecoder (default 6 layers) cross-attending to those anchors.
  - State (proprio) is projected and appended to each anchor as the last token, so the
    expert sees it via cross-attention.
  - Flow-matching action loss (MSE on velocity, Beta(1.5, 1.0) timestep schedule).
"""
from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues
from .action_expert import ActionExpertDecoder
from .configuration_sawseenvla_qwen import SawSeenVLAQwenConfig
from .qwen_encoder import QwenEncoder

OBS_TASK = "task"


def _pad_vector(vec: Tensor, new_dim: int) -> Tensor:
    if vec.shape[-1] == new_dim:
        return vec
    pad = new_dim - vec.shape[-1]
    if pad < 0:
        return vec[..., :new_dim]
    return F.pad(vec, (0, pad))


class SawSeenVLAQwenModel(nn.Module):
    """Flow-matching model: prefix encoded by Qwen, velocity decoded by expert."""

    def __init__(self, config: SawSeenVLAQwenConfig):
        super().__init__()
        self.config = config

        self.encoder = QwenEncoder(
            model_name=config.vlm_model_name,
            image_size=config.image_size,
            freeze=config.freeze_vlm,
            load_weights=config.load_vlm_weights,
            tokenizer_max_length=config.tokenizer_max_length,
            pad_language_to=config.pad_language_to,
        )

        self.expert = ActionExpertDecoder(
            action_dim=config.max_action_dim,
            chunk_size=config.chunk_size,
            n_layers=len(config.qwen_cross_attn_layer_indices),
            hidden=config.expert_hidden_size,
            n_heads=config.expert_num_heads,
            intermediate=config.expert_intermediate_size,
            vlm_hidden=self.encoder.vlm_hidden_size,
            dropout=config.expert_dropout,
            min_period=config.min_period,
            max_period=config.max_period,
        )

        # Robot proprio → Qwen-hidden-size, prepended to each anchor as an extra token
        # so the expert sees state via cross-attention. We project ONCE and reuse for
        # every anchor (cheap; the state is the same regardless of layer depth).
        self.state_proj = nn.Linear(config.max_state_dim, self.encoder.vlm_hidden_size)
        if not config.train_state_proj:
            for p in self.state_proj.parameters():
                p.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the EXPERT (trainable) sub-module. Qwen may be in bf16 while the
        expert stays in fp32; we cast inputs to the expert's dtype, not Qwen's."""
        return next(p for p in self.expert.parameters() if p.is_floating_point()).dtype

    def encode_prefix(
        self,
        images_per_cam: list[Tensor],
        tasks: list[str],
        state: Tensor,
    ) -> tuple[list[Tensor], Tensor]:
        """Run Qwen once and return per-anchor hidden states + attention mask, with
        a state token appended to each anchor."""
        anchors, attn_mask = self.encoder(
            images_per_cam, tasks, anchor_layer_indices=self.config.qwen_cross_attn_layer_indices
        )
        # state: (B, max_state_dim) → (B, 1, vlm_hidden)
        state_tok = self.state_proj(state.to(anchors[0].dtype)).unsqueeze(1)
        anchors_with_state = [torch.cat([a, state_tok], dim=1) for a in anchors]
        # Extend mask by one valid token for the state.
        ones = torch.ones(attn_mask.shape[0], 1, device=attn_mask.device, dtype=attn_mask.dtype)
        attn_mask = torch.cat([attn_mask, ones], dim=1)
        return anchors_with_state, attn_mask

    def forward(
        self,
        images_per_cam: list[Tensor],
        tasks: list[str],
        state: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Training forward: returns per-(batch, time, action) MSE loss (B, chunk, action_dim)."""
        B = state.shape[0]
        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            beta_dist = torch.distributions.Beta(
                concentration1=torch.tensor(1.5, device=actions.device),
                concentration0=torch.tensor(1.0, device=actions.device),
            )
            time = beta_dist.sample((B,)) * 0.999 + 0.001

        t = time[:, None, None]
        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions

        anchors, prefix_mask = self.encode_prefix(images_per_cam, tasks, state)
        v_t = self.expert(x_t.to(self.dtype), time.to(self.dtype), anchors, prefix_mask)
        losses = F.mse_loss(v_t.float(), u_t.float(), reduction="none")
        return losses

    @torch.no_grad()
    def sample_actions(
        self,
        images_per_cam: list[Tensor],
        tasks: list[str],
        state: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Flow-matching denoising loop. Runs Qwen ONCE, expert K times."""
        K = self.config.num_steps
        B = state.shape[0]
        action_dim = self.config.max_action_dim
        chunk = self.config.chunk_size

        anchors, prefix_mask = self.encode_prefix(images_per_cam, tasks, state)
        x_t = noise if noise is not None else torch.randn(B, chunk, action_dim, device=state.device)
        x_t = x_t.to(self.dtype)
        dt = 1.0 / K
        for step in range(K):
            t = torch.full((B,), 1.0 - step * dt, device=state.device, dtype=self.dtype)
            v_t = self.expert(x_t, t, anchors, prefix_mask)
            x_t = x_t - dt * v_t
        return x_t


class SawSeenVLAQwenPolicy(PreTrainedPolicy):
    config_class = SawSeenVLAQwenConfig
    name = "sawseenvla_qwen"

    def __init__(self, config: SawSeenVLAQwenConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = SawSeenVLAQwenModel(config)
        self.reset()

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def get_optim_params(self) -> dict:
        # Train everything that has requires_grad=True (encoder is frozen via param-level flag).
        return [p for p in self.parameters() if p.requires_grad]

    def _extract_tasks(self, batch: dict[str, Any]) -> list[str]:
        # The standard processor pipeline passes "task" as either a list of strings
        # (per-sample) or a single string. Normalize to list[str] of length B.
        t = batch.get(OBS_TASK) or batch.get("task")
        if t is None:
            raise KeyError(
                "Expected 'task' string(s) in batch. Make sure the dataset provides per-sample "
                "task instructions and that the processor pipeline preserves them."
            )
        if isinstance(t, str):
            t = [t]
        if isinstance(t, (list, tuple)):
            return list(t)
        raise TypeError(f"Unsupported task type: {type(t)}")

    def _prepare_images(self, batch: dict[str, Any]) -> list[Tensor]:
        """Return list of (B, 3, H, W) tensors, one per camera. Accepts (B, T, C, H, W)
        (takes the last time step) or (B, C, H, W)."""
        out = []
        for key in self.config.image_features:
            if key not in batch:
                continue
            img = batch[key]
            if img.ndim == 5:
                img = img[:, -1, :, :, :]
            out.append(img)
        if not out:
            raise KeyError(
                f"No image features present in batch. Expected one of {list(self.config.image_features)}"
            )
        return out

    def _prepare_state(self, batch: dict[str, Any]) -> Tensor:
        state = batch[OBS_STATE]
        if state.ndim > 2:
            state = state[:, -1, :]
        return _pad_vector(state, self.config.max_state_dim)

    def _prepare_actions(self, batch: dict[str, Any]) -> Tensor:
        return _pad_vector(batch[ACTION], self.config.max_action_dim)

    def forward(self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"):
        images_per_cam = self._prepare_images(batch)
        tasks = self._extract_tasks(batch)
        state = self._prepare_state(batch)
        actions = self._prepare_actions(batch)
        actions_is_pad = batch.get("action_is_pad")

        losses = self.model.forward(images_per_cam, tasks, state, actions, noise=noise, time=time)
        loss_dict = {"losses_after_forward": losses.detach().mean().item()}

        if actions_is_pad is not None:
            in_bound = ~actions_is_pad
            losses = losses * in_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.detach().mean().item()

        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.detach().mean().item()

        if reduction == "none":
            if actions_is_pad is None:
                per_sample = losses.mean(dim=(1, 2))
            else:
                num_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
                per_sample = losses.sum(dim=(1, 2)) / num_valid
            loss_dict["loss"] = per_sample.mean().item()
            return per_sample, loss_dict
        else:
            if actions_is_pad is None:
                loss = losses.mean()
            else:
                num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
                loss = losses.sum() / num_valid
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        return self._get_action_chunk(batch, noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None) -> Tensor:
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)
        images_per_cam = self._prepare_images(batch)
        tasks = self._extract_tasks(batch)
        state = self._prepare_state(batch)
        actions = self.model.sample_actions(images_per_cam, tasks, state, noise=noise)
        original_action_dim = self.config.action_feature.shape[0]
        return actions[:, :, :original_action_dim]
