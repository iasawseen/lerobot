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

    def _state_to_lewm_proprio(self, state: Tensor | None) -> Tensor | None:
        """Slice the policy's padded state down to the JEPA's training
        ``proprio_dim`` (8 by default for LIBERO).

        The policy carries state padded to ``max_state_dim=32``; the JEPA's
        ``proprio_enc`` was trained at the dataset's *true* proprio width
        (typically 8 = 7 joint + 1 gripper). Returns None if ``state`` is
        None — caller decides whether to forward None (legacy ckpt, OK)
        or raise (proprio-in-state ckpt, missing input).
        """
        if state is None:
            return None
        p = self.config.lewm_proprio_dim
        if state.shape[-1] < p:
            raise ValueError(
                f"state has dim {state.shape[-1]} < lewm_proprio_dim={p}; "
                f"the policy's state pad width is too narrow for the JEPA "
                f"proprio it was trained on."
            )
        return state[..., :p]

    def compute_lewm_tokens(
        self, images: list[Tensor], state: Tensor | None = None
    ) -> Tensor | None:
        """Per-camera CLS tokens projected to the action expert's hidden size.

        Returns ``(B, n_cam, expert_hidden)`` (or ``(B, K, expert_hidden)``
        for multi-token Plan B). The action expert's suffix prepends these
        tokens with bidirectional attention (one block, ``att_mask=[1, 0,
        ..., 0]``) — same wiring as the parent's single-cam case, but the
        token count is now per-camera rather than per-patch-slice.

        ``state`` is sliced to ``lewm_proprio_dim`` and forwarded to the
        encoder for proprio-in-state checkpoints; ignored otherwise.
        """
        if self.lewm_encoder is None or self.lewm_proj is None:
            return None
        img_dict = self._images_to_lewm_dict(images)
        proprio = self._state_to_lewm_proprio(state)
        tokens = self.lewm_encoder.encode_tokens(img_dict, proprio=proprio)
        return self.lewm_proj(tokens.to(self.lewm_proj.weight.dtype))

    def _encode_lewm_emb(
        self, images: list[Tensor], state: Tensor | None = None
    ) -> Tensor:
        """Post-projector latent per batch element — used as LGE anchor
        and LGE regression target.

        Returns:
            * ``(B, D)`` for legacy single-token JEPA (always).
            * ``(B, K, D)`` for multi-token JEPA when
              ``latent_goal_residual=True`` — the residual LGE path
              keeps the K axis end-to-end (per le-wm's Plan-B SIGReg
              convention).
            * ``(B, K*D)`` for multi-token JEPA when
              ``latent_goal_residual=False`` — legacy compatibility:
              flatten K*D so downstream code that expects 2D
              ``z_t / z_g`` still works.

        ``state`` is sliced to ``lewm_proprio_dim`` and forwarded to the
        encoder for proprio-in-state checkpoints; ignored otherwise.
        """
        if self.lewm_encoder is None:
            raise RuntimeError(
                "_encode_lewm_emb called without a loaded lewm encoder. "
                "Set lewm_encoder_path in the config."
            )
        img_dict = self._images_to_lewm_dict(images)
        proprio = self._state_to_lewm_proprio(state)
        with torch.no_grad():
            emb = self.lewm_encoder.encode_cls(img_dict, proprio=proprio)  # (B, D) or (B, K, D)

        # Residual LGE path keeps the K axis native — the LGE expert
        # processes a length-K sequence, in/out projectors broadcast
        # over K via nn.Linear's last-dim contract, SIGReg follows
        # le-wm's Plan-B "(b k d) -> 1 (b k) d" pattern. Legacy
        # non-residual path collapses K into D for compat with the
        # original (B, D) LGE projectors.
        if emb.ndim == 3 and not self.config.latent_goal_residual:
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

        Three dispatch paths (``self.config.mpc_horizon_mode``):

          * ``"single"`` (default) — single-shot var-stride prediction at
            ``k_tail = latent_goal_offset``. One ``predict()`` call per
            candidate; the predicted state at +k_tail is compared (MSE)
            to z_g (also at +k_tail). Fastest path; requires k_tail in
            the checkpoint's trained ``k_choices``.

          * ``"ar"`` — autoregressive rollout at k=1 for ``k_tail``
            steps. ``predict()`` called k_tail times. Slower but works
            when the checkpoint wasn't trained at the desired k_tail.

          * ``"multi_offset"`` — for each ``k`` in
            ``self.config.mpc_offsets``, one var-stride single-shot
            prediction at k_tail=k, MSE-to-z_g multiplied by the
            corresponding weight; summed across offsets.

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
        mode = self.config.mpc_horizon_mode
        if mode == "multi_offset":
            return self._rollout_multi_offset_impl(
                z_t_emb, z_g_emb, candidates,
                list(self.config.mpc_offsets),
                list(self.config.mpc_offset_weights),
            )
        if mode == "ar":
            return self._rollout_ar(z_t_emb, z_g_emb, candidates)
        # "single" — single-shot at k_tail = latent_goal_offset
        k = self.config.latent_goal_offset
        return self._rollout_multi_offset_impl(
            z_t_emb, z_g_emb, candidates, [k], [1.0],
        )

    def _action_buffer_zeros(self, B_eff: int, T_slots: int, A: int, device, dtype) -> Tensor:
        """Empty ``(B_eff, T_slots, k_max, A)`` action tensor."""
        k_max = self.config.mpc_action_k_max
        return torch.zeros(B_eff, T_slots, k_max, A, device=device, dtype=dtype)

    def _per_slot_k(self, B_eff: int, k_list: list[int], device) -> Tensor:
        """Per-slot k tensor of shape ``(B_eff, len(k_list))``."""
        k = torch.tensor(k_list, device=device, dtype=torch.long)
        return k.view(1, -1).expand(B_eff, -1).contiguous()

    def _rollout_ar(
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
        z_t_flat = z_t_emb.repeat_interleave(N, dim=0)  # (B*N, D) or (B*N, K, D)
        if z_t_flat.ndim == 2:
            # Single-token JEPA: (B*N, D) → (B*N, HS, D).
            emb_buf = z_t_flat.unsqueeze(1).expand(-1, HS, -1).contiguous()
        elif z_t_flat.ndim == 3:
            # Multi-token JEPA (Plan B): (B*N, K, D) → (B*N, HS, K, D).
            # The predictor keeps the K axis (jepa.py:517); residual-LGE
            # checkpoints carry z_t through as (B, K, D) end-to-end.
            emb_buf = z_t_flat.unsqueeze(1).expand(-1, HS, -1, -1).contiguous()
        else:
            raise RuntimeError(f"unexpected z_t ndim {z_t_flat.ndim}; want 2 or 3")

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

        z_final = emb_buf[:, -1].to(torch.float32)  # (B*N, D) or (B*N, K, D)
        z_g_flat = z_g_emb.repeat_interleave(N, dim=0).to(torch.float32)
        cost = ((z_final - z_g_flat) ** 2).flatten(1).sum(dim=-1)  # (B*N,)
        return cost.view(B, N)

    def _rollout_multi_offset_impl(
        self,
        z_t_emb: Tensor,
        z_g_emb: Tensor,
        candidates: Tensor,
        offsets: list[int],
        weights: list[float],
    ) -> Tensor:
        """For each k in ``offsets``, single-shot var-stride prediction
        at k_tail=k from the fabricated 3-slot history. Cost is the
        weighted sum of per-offset MSE-to-z_g (same goal across offsets).

        Slot k assignment:
          * slot 0, 1 → fabricated past, k=1, action position 0 = zero
          * slot 2 → candidate-future, k=k_tail, action positions 0..k-1
                    hold the candidate's first k actions.

        One ``predict_step`` per offset; cheap relative to AR rollouts of
        length T. Called by the dispatcher for both ``"single"`` mode
        (offsets=[latent_goal_offset], weights=[1.0]) and
        ``"multi_offset"`` mode (caller-provided offsets/weights).
        """
        assert self.lewm_world is not None
        B, N, T, A = candidates.shape
        HS = self.lewm_world.history_size  # = 3
        device = candidates.device
        dtype = candidates.dtype
        flat = candidates.reshape(B * N, T, A)

        # History embedding: repeat z_t HS times (same fabrication as AR
        # path). Single-token z_t is (B*N, D) → (B*N, HS, D); multi-token
        # (Plan B) z_t is (B*N, K, D) → (B*N, HS, K, D). The predictor
        # infers token-ness from emb.ndim (jepa.py:517), and
        # encode_actions/predict_step are shape-agnostic to K.
        z_t_flat = z_t_emb.repeat_interleave(N, dim=0)
        if z_t_flat.ndim == 2:
            emb_buf = z_t_flat.unsqueeze(1).expand(-1, HS, -1).contiguous()  # (B*N, HS, D)
        elif z_t_flat.ndim == 3:
            emb_buf = z_t_flat.unsqueeze(1).expand(-1, HS, -1, -1).contiguous()  # (B*N, HS, K, D)
        else:
            raise RuntimeError(f"unexpected z_t ndim {z_t_flat.ndim}; want 2 (single) or 3 (multi-token)")
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
            # pred[:, -1] is the +k_tail state: (B*N, D) single-token or
            # (B*N, K, D) multi-token. flatten(1) collapses the trailing
            # (K, D) → (K*D) so the squared error sums over all tokens.
            z_pred = pred[:, -1].to(torch.float32)
            sqerr = ((z_pred - z_g_flat) ** 2).flatten(1).sum(dim=-1)  # (B*N,)
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
