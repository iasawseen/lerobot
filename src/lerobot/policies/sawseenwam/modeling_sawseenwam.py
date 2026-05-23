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

"""SawSeenWAM — SawSeenVLAWM with the new dual-encoder / variable-stride le-wm.

Subclasses the existing ``VLAFlowMatching`` / ``SawSeenVLAWMPolicy`` and
only overrides the four le-wm-touching seams:

* ``_load_lewm_encoder`` — load ``LeWMDualVisionEncoder`` from the new
  pickled JEPA (``encoders: nn.ModuleList`` + ``projector`` +
  optional ``query_reducers``).
* ``_load_lewm_world`` — same checkpoint, but expose the predictor +
  action_encoder + pred_proj sub-modules for MPC rollouts.
* ``compute_lewm_tokens`` — feed each camera image to its own encoder
  (no horizontal concat) and return ``(B, n_cam, expert_hidden)`` tokens.
* ``_encode_lewm_emb`` — same dual-cam pathway, but returns the (B, D)
  fused post-projector latent used as LGE anchor / target.
* ``_lewm_rollout_score`` — predict candidate chunks with the new
  variable-stride ``action_encoder`` (per-slot ``k``). Two horizon modes
  (config ``mpc_horizon_mode``):
    - ``"single"`` — AR rollout at k=1 for ``chunk_size`` steps, MSE to z_g.
    - ``"multi_offset"`` — for each ``k`` in ``mpc_offsets``, run a single-shot
      var-stride prediction at ``k_tail=k`` and weighted-sum MSE to the
      same z_g.

The rest of the model (action expert, LGE, Mode 3, SIGReg, MPC schemes,
iCEM β, score-floor, MPPI / CEM dispatcher) is inherited unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from lerobot.utils.import_utils import require_package

from ..sawseenvlawm.modeling_sawseenvlawm import (
    SawSeenVLAWMPolicy,
    VLAFlowMatching,
)
from .configuration_sawseenwam import SawSeenWAMConfig
from .lewm_dual_encoder import LeWMDualVisionEncoder, LeWMDualWorldModel

if TYPE_CHECKING:
    from ..sawseenvlawm.modeling_sawseenvlawm import RTCProcessor


class WAMFlowMatching(VLAFlowMatching):
    """Inner model — same dual-expert SmolVLM + LGE + MPC dispatcher,
    with le-wm pathways swapped to the new dual-encoder JEPA."""

    config: SawSeenWAMConfig  # narrowed type for the IDE

    def __init__(self, config: SawSeenWAMConfig, rtc_processor: "RTCProcessor | None" = None):
        # Parent __init__ calls our overridden _load_lewm_encoder /
        # _load_lewm_world (the hooks were extracted exactly for this).
        super().__init__(config, rtc_processor=rtc_processor)

    # ── lewm load hooks (overrides parent) ───────────────────────────

    def _load_lewm_encoder(self) -> None:
        """Load the new-le-wm dual-encoder vision wrapper + action-expert proj."""
        if self.config.lewm_encoder_path is None:
            return
        if self.config.lewm_inject_to not in ("suffix", "none"):
            raise ValueError(
                f"lewm_inject_to must be 'suffix' or 'none'; got {self.config.lewm_inject_to!r}"
            )

        # Load the full JEPA once; the vision encoder holds a reference to
        # it (it's the same pickled module that will also back MPC's
        # predictor rollouts).
        self._lewm_world_loaded = LeWMDualWorldModel.from_lewm_checkpoint(
            self.config.lewm_encoder_path,
            pixel_keys=self.config.lewm_pixel_keys,
            image_height=self.config.lewm_image_height,
            image_width=self.config.lewm_image_width,
            freeze=self.config.lewm_freeze,
        )
        self.lewm_encoder = LeWMDualVisionEncoder(self._lewm_world_loaded)  # type: ignore[assignment]

        if self.config.lewm_inject_to == "suffix":
            proj_out = self.vlm_with_expert.expert_hidden_size
            self.lewm_proj = nn.Linear(self.lewm_encoder.output_dim, proj_out)

    def _load_lewm_world(self) -> None:
        """For MPC, reuse the same loaded JEPA (no double-load)."""
        if not self.config.mpc_enabled:
            return
        # If the encoder was loaded, reuse its JEPA — same pickle, both
        # consumers see the same parameters. Otherwise load fresh from
        # ``mpc_predictor_path``.
        existing = getattr(self, "_lewm_world_loaded", None)
        if existing is not None and (
            self.config.mpc_predictor_path is None
            or self.config.mpc_predictor_path == self.config.lewm_encoder_path
        ):
            self.lewm_world = existing
            return

        predictor_path = self.config.mpc_predictor_path or self.config.lewm_encoder_path
        self.lewm_world = LeWMDualWorldModel.from_lewm_checkpoint(
            predictor_path,
            pixel_keys=self.config.lewm_pixel_keys,
            image_height=self.config.lewm_image_height,
            image_width=self.config.lewm_image_width,
            freeze=True,
        )

    # ── image → lewm side-channel (overrides parent) ─────────────────

    def _images_to_lewm_dict(self, images: list[Tensor]) -> dict[str, Tensor]:
        """Map an ordered list of camera tensors to the lewm pixel_keys dict.

        The parent's ``prepare_images`` returns ``images`` as a list in the
        order of the dataset's ``OBS_IMAGES.*`` keys. SawSeenWAM expects
        the *first* N entries to correspond to ``config.lewm_pixel_keys`` by
        index. Extra images (e.g. empty_cameras pad slots) are ignored —
        only the lewm-bound cams flow into the dual encoder.
        """
        keys = self.config.lewm_pixel_keys
        if len(images) < len(keys):
            raise ValueError(
                f"prepare_images produced {len(images)} streams but lewm_pixel_keys "
                f"requires {len(keys)} ({keys})"
            )
        return {keys[i]: images[i] for i in range(len(keys))}

    def compute_lewm_tokens(self, images: list[Tensor]) -> Tensor | None:
        """Per-camera CLS tokens projected to the action expert's hidden size.

        Returns ``(B, n_cam, expert_hidden)`` (or ``(B, K, expert_hidden)``
        for multi-token Plan B). The action expert's suffix prepends these
        tokens with bidirectional attention (one block, ``att_mask=[1, 0,
        ..., 0]``) — same wiring as the parent's single-cam case, but the
        token count is now per-camera rather than per-patch-slice.
        """
        if self.lewm_encoder is None or self.lewm_proj is None:
            return None
        img_dict = self._images_to_lewm_dict(images)
        tokens = self.lewm_encoder.encode_tokens(img_dict)  # (B, n_cam, 192) or (B, K, 192)
        return self.lewm_proj(tokens.to(self.lewm_proj.weight.dtype))

    def _encode_lewm_emb(self, images: list[Tensor]) -> Tensor:
        """Single post-projector latent per batch element (B, D) — used as
        LGE anchor (current frame) and LGE regression target (chunk-end).
        """
        if self.lewm_encoder is None:
            raise RuntimeError(
                "_encode_lewm_emb called without a loaded lewm encoder. "
                "Set lewm_encoder_path in the config."
            )
        img_dict = self._images_to_lewm_dict(images)
        with torch.no_grad():
            emb = self.lewm_encoder.encode_cls(img_dict)  # (B, D) or (B, K, D)

        # LGE supervises against a (B, D) target — if the JEPA is in
        # multi-token Plan B mode, flatten the K axis into D so LGE sees
        # a single vector. Keep this path explicit so configs that
        # accidentally pair lewm_multi_token=True with a checkpoint that
        # doesn't have query_reducers raise at load (in
        # LeWMDualWorldModel.from_lewm_checkpoint), not silently here.
        if emb.ndim == 3:
            B, K, D = emb.shape
            emb = emb.reshape(B, K * D)
        return emb

    # ── MPC rollout (overrides parent) ───────────────────────────────

    def _lewm_rollout_score(
        self,
        z_t_emb: Tensor,
        z_g_emb: Tensor,
        candidates: Tensor,
    ) -> Tensor:
        """Roll candidates through the new dual-encoder JEPA's predictor.

        Two dispatch paths (``self.config.mpc_horizon_mode``):

          * ``"single"`` — AR rollout at k=1 for ``chunk_size`` steps,
            matching SawSeenVLAWM behavior under the new action_encoder API.

          * ``"multi_offset"`` — for each ``k`` in ``self.config.mpc_offsets``,
            one var-stride single-shot prediction at k_tail=k, MSE-to-z_g
            multiplied by the corresponding weight; summed across offsets.

        Args:
            z_t_emb: ``(B, D)`` current latent in JEPA-projector space.
                If the JEPA is multi-token (Plan B), this comes in already
                flattened from (B, K, D) by ``_encode_lewm_emb``.
            z_g_emb: ``(B, D)`` goal latent. Same flattening as z_t.
            candidates: ``(B, N, T, A_raw)`` candidate action chunks.

        Returns:
            ``cost``: ``(B, N)`` per-candidate cost.
        """
        assert self.lewm_world is not None, "MPC enabled without lewm_world loaded"
        if self.config.mpc_horizon_mode == "multi_offset":
            return self._rollout_multi_offset(z_t_emb, z_g_emb, candidates)
        return self._rollout_single_ar(z_t_emb, z_g_emb, candidates)

    def _action_buffer_zeros(self, B_eff: int, T_slots: int, A: int, device, dtype) -> Tensor:
        """Empty ``(B_eff, T_slots, k_max, A)`` action tensor."""
        k_max = self.config.mpc_action_k_max
        return torch.zeros(B_eff, T_slots, k_max, A, device=device, dtype=dtype)

    def _per_slot_k(self, B_eff: int, k_list: list[int], device) -> Tensor:
        """Per-slot k tensor of shape ``(B_eff, len(k_list))``."""
        k = torch.tensor(k_list, device=device, dtype=torch.long)
        return k.view(1, -1).expand(B_eff, -1).contiguous()

    def _rollout_single_ar(
        self,
        z_t_emb: Tensor,
        z_g_emb: Tensor,
        candidates: Tensor,
    ) -> Tensor:
        """AR rollout at k=1 for ``T_rollout`` steps, MSE to z_g.

        ``T_rollout = config.latent_goal_offset`` (= ``chunk_size`` when
        ``latent_goal_target_offset`` is unset, for backward compat).
        Candidates have T=chunk_size actions; we only use the first
        T_rollout — the remaining tail isn't scored (and typically isn't
        executed either since ``n_action_steps`` < ``chunk_size``).

        History fabrication: the new JEPA still uses history_size=3 by
        default. We have one real frame (z_t), so we repeat it HS times
        (mirroring SawSeenVLAWM's behavior). Per-slot k = 1 everywhere —
        the action_encoder treats every slot as a 1-step stride.
        """
        assert self.lewm_world is not None
        B, N, T, A = candidates.shape
        T_rollout = self.config.latent_goal_offset
        if T_rollout > T:
            raise RuntimeError(
                f"latent_goal_offset={T_rollout} exceeds candidate chunk T={T}; "
                "this should have been caught by config validation."
            )
        HS = self.lewm_world.history_size
        device = candidates.device
        dtype = candidates.dtype
        # Slice the first T_rollout actions from each candidate.
        flat = candidates[:, :, :T_rollout, :].reshape(B * N, T_rollout, A)

        # Initial history embedding: repeat z_t HS times.
        z_t_flat = z_t_emb.repeat_interleave(N, dim=0)  # (B*N, D) or (B*N, K*D)
        if z_t_flat.ndim == 2:
            # Single-token JEPA: (B*N, D) → (B*N, HS, D).
            emb_buf = z_t_flat.unsqueeze(1).expand(-1, HS, -1).contiguous()
        else:
            # Multi-token JEPA path (Plan B): the LGE was supervised on
            # the flattened latent, but the predictor still works in the
            # (B, T, K, D) shape. Reshape back.
            B_eff, KD = z_t_flat.shape
            K = self.lewm_world.jepa.encoders[0].config.hidden_size  # placeholder
            raise NotImplementedError(
                "multi-token (Plan B) MPC rollout not yet wired — emb_buf "
                "needs to be (B*N, HS, K, D) but z_t arrives flattened."
            )

        # Past-action slots (HS-1 fabricated zero actions).
        past_act = torch.zeros(B * N, HS - 1, A, device=device, dtype=dtype)
        # Append candidate actions one at a time. Per-step we feed
        # (HS-1 fabricated + 1 current) = HS slots into the action encoder.
        act_history = past_act  # (B*N, HS-1, A) → grows over the loop

        for t in range(T_rollout):
            # Extend act_history with this step's action.
            act_step = flat[:, t : t + 1, :]  # (B*N, 1, A)
            act_history = torch.cat([act_history, act_step], dim=1)

            # Take last HS slots, pack into (B*N, HS, k_max, A) with k=1 each.
            act_win = act_history[:, -HS:]  # (B*N, HS, A)
            act_packed = self._action_buffer_zeros(B * N, HS, A, device, dtype)
            act_packed[:, :, 0, :] = act_win
            k_per_slot = self._per_slot_k(B * N, [1] * HS, device)

            emb_win = emb_buf[:, -HS:]  # (B*N, HS, D)
            act_emb, action_tokens = self.lewm_world.encode_actions(act_packed, k_per_slot)
            pred = self.lewm_world.predict_step(emb_win, act_emb, action_tokens=action_tokens)
            # pred[:, -1] is the prediction at the last slot's stride
            # (k=1, so "one step ahead"). Append to emb_buf.
            next_emb = pred[:, -1:, :].to(emb_buf.dtype)
            emb_buf = torch.cat([emb_buf, next_emb], dim=1)

        z_final = emb_buf[:, -1].to(torch.float32)  # (B*N, D)
        z_g_flat = z_g_emb.repeat_interleave(N, dim=0).to(torch.float32)
        cost = ((z_final - z_g_flat) ** 2).sum(dim=-1)  # (B*N,)
        return cost.view(B, N)

    def _rollout_multi_offset(
        self,
        z_t_emb: Tensor,
        z_g_emb: Tensor,
        candidates: Tensor,
    ) -> Tensor:
        """For each k in mpc_offsets, single-shot var-stride prediction
        at k_tail=k from the fabricated 3-slot history. Cost is the
        weighted sum of per-offset MSE-to-z_g (same goal across offsets).

        Slot k assignment:
          * slot 0, 1 → fabricated past, k=1, action position 0 = zero
          * slot 2 → candidate-future, k=k_tail, action positions 0..k-1
                    hold the candidate's first k actions.

        One ``predict_step`` per offset; cheap relative to AR rollouts of
        length T.
        """
        assert self.lewm_world is not None
        B, N, T, A = candidates.shape
        HS = self.lewm_world.history_size  # = 3
        device = candidates.device
        dtype = candidates.dtype
        flat = candidates.reshape(B * N, T, A)
        offsets = list(self.config.mpc_offsets)
        weights = list(self.config.mpc_offset_weights)

        # History embedding: repeat z_t HS times (same fabrication as AR
        # path).
        z_t_flat = z_t_emb.repeat_interleave(N, dim=0)
        if z_t_flat.ndim != 2:
            raise NotImplementedError(
                "multi-token (Plan B) MPC rollout not yet wired for multi_offset."
            )
        emb_buf = z_t_flat.unsqueeze(1).expand(-1, HS, -1).contiguous()  # (B*N, 3, D)
        z_g_flat = z_g_emb.repeat_interleave(N, dim=0).to(torch.float32)

        total_cost = torch.zeros(B * N, device=device, dtype=torch.float32)

        for k_tail, w in zip(offsets, weights):
            # Build per-slot action: slots 0/1 zero, slot 2 holds first k_tail
            # actions of the candidate.
            act_packed = self._action_buffer_zeros(B * N, HS, A, device, dtype)
            # Slot 2 of size [k_tail x A]; check chunk has enough actions.
            if k_tail > T:
                raise RuntimeError(
                    f"mpc_offsets requires k_tail={k_tail} actions but candidate "
                    f"chunk only has T={T}. Increase chunk_size or shorten offsets."
                )
            act_packed[:, HS - 1, :k_tail, :] = flat[:, :k_tail, :]
            k_per_slot = self._per_slot_k(B * N, [1] * (HS - 1) + [k_tail], device)

            act_emb, action_tokens = self.lewm_world.encode_actions(act_packed, k_per_slot)
            pred = self.lewm_world.predict_step(emb_buf, act_emb, action_tokens=action_tokens)
            z_pred = pred[:, -1].to(torch.float32)  # (B*N, D) — state at +k_tail
            sqerr = ((z_pred - z_g_flat) ** 2).sum(dim=-1)
            total_cost = total_cost + float(w) * sqerr

        return total_cost.view(B, N)


class SawSeenWAMPolicy(SawSeenVLAWMPolicy):
    """Wrapper around ``WAMFlowMatching`` — same lerobot processor /
    queue plumbing as SawSeenVLAWM, only the inner model differs.
    """

    config_class = SawSeenWAMConfig
    name = "sawseenwam"

    def __init__(self, config: SawSeenWAMConfig, **kwargs):
        require_package("transformers", extra="smolvla")
        # Skip the immediate parent's __init__ — it creates
        # VLAFlowMatching; we need WAMFlowMatching instead. Go up one
        # more level to PreTrainedPolicy.__init__ via super().
        # Equivalent to SawSeenVLAWMPolicy.__init__ but with a different
        # ``self.model`` constructor.
        super(SawSeenVLAWMPolicy, self).__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = WAMFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()
