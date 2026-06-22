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

"""Configuration for SawSeenWAM — SawSeenVLAWM modified for the new le-wm.

Two material differences from ``SawSeenVLAWMConfig``:

1. **Dual-encoder lewm.** The new le-wm at
   ``/home/lucius/data/personal-hive/code/le-wm`` ships
   ``encoders: nn.ModuleList`` with one ViT per camera (``pixel_keys =
   ("pixels", "pixels_wrist")``) rather than the old single ViT over a
   camera-concat 224×448 image. ``lewm_image_width`` therefore defaults to
   224 (per-camera, square) instead of 448, and a new ``lewm_pixel_keys``
   field declares which input image streams flow to which encoder slot.

2. **Variable-stride / multi-offset cost.** The new JEPA exposes
   ``get_cost_var_stride*`` methods that score predictions at multiple
   horizons. Wired here as ``mpc_horizon_mode ∈ {"single", "multi_offset"}``
   plus ``mpc_offsets`` / ``mpc_offset_weights`` for the multi-offset case.
   Default ``"single"`` preserves the legacy AR-rollout-to-chunk-end MPC.

Other defaults (LGE, Mode 3, SIGReg, MPC schemes, iCEM β, score-floor)
are inherited verbatim — same flow-matching action expert, same LGE in
projector space, same MPC dispatcher.
"""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("sawseenwam")
@dataclass
class SawSeenWAMConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    max_state_dim: int = 32
    max_action_dim: int = 32

    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    empty_cameras: int = 0
    adapt_to_pi_aloha: bool = False
    use_delta_joint_actions_aloha: bool = False

    tokenizer_max_length: int = 48
    num_steps: int = 10
    use_cache: bool = True

    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = False
    add_image_special_tokens: bool = False
    attention_mode: str = "cross_attn"
    prefix_length: int = -1
    pad_language_to: str = "longest"
    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75

    min_period: float = 4e-3
    max_period: float = 4.0

    rtc_config: RTCConfig | None = None

    compile_model: bool = False
    compile_mode: str = "max-autotune"

    # ── new-le-wm dual-encoder side-channel ──────────────────────────
    # Path to the new le-wm ``<name>_object.ckpt`` pickle (full JEPA with
    # ``encoders: nn.ModuleList`` and ``projector`` / ``pred_proj``).
    lewm_encoder_path: str | None = None
    lewm_freeze: bool = True
    # Number of tokens per camera to expose to the action expert.
    # ``1`` (default) = CLS-only per cam.
    # For multi-token (Plan B) Q+1 checkpoints, set to the per-cam token
    # count baked into the checkpoint — the loader cross-checks.
    lewm_num_tokens: int = 1
    # Per-camera ViT input. The new lewm trained each ViT on per-camera
    # 224×224, *not* the old camera-concat 224×448. Keep these square
    # unless you're loading a hand-crafted checkpoint.
    lewm_image_height: int = 224
    lewm_image_width: int = 224
    lewm_patch_size: int = 14
    # Ordered tuple of image keys to feed into the dual encoders. The
    # first entry binds to ``encoders[0]``, second to ``encoders[1]``, and
    # so on. Must match the order baked into the loaded JEPA's
    # ``pixel_keys`` (new-lewm CLAUDE.md explicitly calls this "load-bearing
    # convention"). LIBERO ships ``("pixels", "pixels_wrist")`` =
    # (agentview, eye-in-hand). lerobot dataset keys are
    # ``OBS_IMAGES.image`` / ``OBS_IMAGES.wrist_image``; the policy maps
    # them to lewm's pixel_keys by index.
    lewm_pixel_keys: tuple[str, ...] = ("pixels", "pixels_wrist")
    # Multi-token / Plan B: when True, the checkpoint is expected to
    # carry ``query_reducers`` and emit ``(B, T, K, D)`` per state where
    # K = (Q+1) * n_cam. Off by default; only switch on with a Plan-B
    # trained checkpoint, since shapes propagate through LGE supervision.
    lewm_multi_token: bool = False
    # Where lewm tokens enter the model — same semantics as SawSeenVLAWM.
    lewm_inject_to: str = "suffix"

    # ── Latent Goal Expert (LGE) — identical to SawSeenVLAWM ─────────
    latent_goal_enabled: bool = False
    latent_goal_loss_weight: float = 1.0
    latent_goal_loss_type: str = "bc"
    latent_goal_num_steps: int = 10
    latent_goal_expert_width_multiplier: float = 0.75
    latent_goal_num_expert_layers: int = -1

    # ── LGE target offset (decoupled from chunk_size in SawSeenWAM) ──
    # Frame offset at which LGE is supervised: the dataset returns
    # ``image[t + latent_goal_target_offset]`` as the chunk-end frame
    # and LGE regresses to its encoded latent. When None (default), the
    # offset falls back to ``chunk_size`` — same behavior as SawSeenVLAWM.
    #
    # Set this when the new-lewm predictor's largest trained ``k_tail``
    # is < ``chunk_size``. Aligns the LGE z_g target with what the
    # predictor was actually trained to predict at, so MPC single-shot
    # scoring at ``k_tail = latent_goal_target_offset`` is semantically
    # meaningful (final-state MSE against the same horizon).
    #
    # Example: latest new-lewm checkpoint has k_choices=(1,2,5,10,25)
    # and we keep chunk_size=50 (action expert still outputs 50-action
    # chunks). Set ``latent_goal_target_offset=25`` so LGE supervises at
    # +25 frames and ``mpc_offsets=(25,)`` cleanly aligns.
    #
    # Must satisfy ``0 < latent_goal_target_offset <= chunk_size`` (the
    # candidate chunk needs at least this many actions to cover the
    # horizon during AR rollout).
    latent_goal_target_offset: int | None = None

    # ── Mode 3 LGE-conditioned action expert ─────────────────────────
    latent_goal_inject_to_action: bool = False
    latent_goal_inject_z_g_source: str = "encoded"
    latent_goal_inject_schedule_end_step: int = 0
    latent_goal_inject_schedule_start_step: int = 0
    latent_goal_inject_detach: bool = True
    latent_goal_train_num_steps: int = 1

    # ── SIGReg on z_g_pred (LGE clean reconstruction) ────────────────
    latent_goal_sigreg_weight: float = 0.0
    latent_goal_sigreg_knots: int = 17
    latent_goal_sigreg_num_proj: int = 1024

    # ── Phase B / MPC inference ──────────────────────────────────────
    mpc_enabled: bool = False
    mpc_scheme: str = "anchor_perturb"  # anchor_perturb | cem | mppi
    mpc_num_candidates: int = 16
    mpc_noise_scale: float = 0.1
    mpc_cem_num_iter: int = 4
    mpc_cem_topk: int = 4
    mpc_cem_anchor_blend: float = 0.5
    mpc_cem_include_anchor: bool = True
    mpc_cem_init_mean: str = "anchor"  # anchor | zero
    mpc_cem_return: str = "best_ever"  # best_ever | final_mean
    mpc_mppi_temperature: float = 1.0
    mpc_mppi_num_iter: int = 4
    mpc_score_floor_margin: float = 0.0
    mpc_icem_beta: float = 0.0
    mpc_predictor_path: str | None = None

    # ── Varied-horizon MPC (new-lewm only) ───────────────────────────
    # Multi-offset cost scoring. New le-wm trains a JEPA that can predict
    # at multiple k_tail offsets (e.g., k∈{1,2,4,8,16}). At inference,
    # candidates can be scored against the LGE-predicted z_g at *each*
    # offset and the weighted-sum cost used.
    #
    # ``mpc_horizon_mode``:
    #   "single"       — *single-shot* var-stride prediction at
    #                    k_tail = ``latent_goal_target_offset`` (= 25 by
    #                    default). One ``predict()`` call per candidate;
    #                    the candidate's first k_tail actions are packed
    #                    into slot 2 of a 3-slot history (per-slot k =
    #                    [1, 1, k_tail]). The predicted state at +k_tail
    #                    is compared (MSE) to the LGE z_g (also at
    #                    +k_tail). Requires k_tail ∈ checkpoint's
    #                    trained k_choices. Recommended default.
    #   "ar"           — Autoregressive rollout at k=1 for k_tail
    #                    steps. ``predict()`` called k_tail times per
    #                    candidate (much slower). Useful if the
    #                    checkpoint wasn't trained at k=k_tail but
    #                    *was* trained at k=1.
    #   "multi_offset" — For each k in ``mpc_offsets``, run a
    #                    single-shot var-stride prediction at k_tail=k,
    #                    weighted-sum MSEs vs the same LGE z_g.
    #                    Generalization of "single" to multiple offsets.
    mpc_horizon_mode: str = "single"
    # k_tail offsets used by ``multi_offset``. Must be a subset of the
    # checkpoint's trained ``k_choices``. The last (max) offset should
    # typically equal chunk_size for parity with the LGE z_g supervision.
    # ``mpc_offsets``: tuple of k_tail values. *Each must be in the
    # checkpoint's trained ``k_choices``* — the action_encoder's k_emb
    # has no entry for unseen k. Latest LIBERO k_choices = (1, 2, 5, 10,
    # 25), so a multi-offset default of (25,) is the largest valid
    # single-shot stride. Note semantic caveat: LGE supervises z_g at
    # ``chunk_size=50`` ahead, but a k_tail=25 single-shot prediction is
    # the JEPA's predicted state at +25 — *not* directly comparable to
    # z_g at +50 unless the cost is interpreted as "directional progress
    # at 25 steps". For chunk-end alignment, prefer ``"single"`` mode
    # (AR k=1 for 50 steps) over ``"multi_offset"``.
    mpc_offsets: tuple[int, ...] = (25,)
    # Weights for each offset (same length as ``mpc_offsets``). Higher
    # weight = stronger contribution to the cost. Not normalized.
    mpc_offset_weights: tuple[float, ...] = (1.0,)
    # k_max baked into the new-lewm action_encoder. Must match the
    # checkpoint's encoder configuration (``encoder.k_max`` after
    # unpickling). Default 25 matches the latest LIBERO multi-offset
    # training (k_choices=(1, 2, 5, 10, 25)). Earlier checkpoints used
    # k_choices=(1, 2, 4, 8, 16) → bump down to 16 if loading those.
    mpc_action_k_max: int = 25

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is not ported in SawSeenWAM."
            )
        if self.latent_goal_inject_to_action and not self.latent_goal_enabled:
            raise ValueError(
                "latent_goal_inject_to_action=True requires latent_goal_enabled=True."
            )
        if self.latent_goal_inject_z_g_source not in ("encoded", "predicted", "scheduled"):
            raise ValueError(
                f"latent_goal_inject_z_g_source must be 'encoded', 'predicted', or "
                f"'scheduled'; got {self.latent_goal_inject_z_g_source!r}"
            )
        if (
            self.latent_goal_inject_z_g_source == "scheduled"
            and self.latent_goal_inject_schedule_end_step <= 0
        ):
            raise ValueError(
                "scheduled source requires latent_goal_inject_schedule_end_step > 0."
            )
        if self.latent_goal_inject_schedule_start_step < 0:
            raise ValueError(
                "latent_goal_inject_schedule_start_step must be ≥ 0."
            )
        if (
            self.latent_goal_inject_z_g_source == "scheduled"
            and self.latent_goal_inject_schedule_start_step
            >= self.latent_goal_inject_schedule_end_step
        ):
            raise ValueError(
                "scheduled requires start_step < end_step."
            )
        if self.latent_goal_train_num_steps < 1:
            raise ValueError("latent_goal_train_num_steps must be ≥ 1.")
        if self.latent_goal_target_offset is not None:
            if self.latent_goal_target_offset < 1:
                raise ValueError(
                    f"latent_goal_target_offset must be ≥ 1; got {self.latent_goal_target_offset}"
                )
            if self.latent_goal_target_offset > self.chunk_size:
                raise ValueError(
                    f"latent_goal_target_offset={self.latent_goal_target_offset} > "
                    f"chunk_size={self.chunk_size} — the candidate action chunk would "
                    f"not have enough actions to reach the target horizon."
                )
        if self.latent_goal_sigreg_weight < 0:
            raise ValueError("latent_goal_sigreg_weight must be ≥ 0.")
        if self.latent_goal_sigreg_weight > 0 and not self.latent_goal_enabled:
            raise ValueError("latent_goal_sigreg_weight > 0 requires latent_goal_enabled.")
        if self.latent_goal_sigreg_knots < 2:
            raise ValueError("latent_goal_sigreg_knots must be ≥ 2.")
        if self.latent_goal_sigreg_num_proj < 1:
            raise ValueError("latent_goal_sigreg_num_proj must be ≥ 1.")

        if len(self.lewm_pixel_keys) < 1:
            raise ValueError("lewm_pixel_keys must be non-empty.")

        if self.mpc_enabled:
            if not self.latent_goal_enabled:
                raise ValueError(
                    "mpc_enabled=True requires latent_goal_enabled=True — LGE z_g is the scoring target."
                )
            if not self.lewm_encoder_path and not self.mpc_predictor_path:
                raise ValueError(
                    "mpc_enabled=True requires a le-wm checkpoint."
                )
            if self.mpc_num_candidates < 2:
                raise ValueError("mpc_num_candidates must be ≥ 2.")
            if self.mpc_scheme not in ("anchor_perturb", "cem", "mppi"):
                raise ValueError(
                    f"mpc_scheme must be 'anchor_perturb', 'cem', or 'mppi'; got {self.mpc_scheme!r}"
                )
            if self.mpc_scheme == "cem":
                if self.mpc_cem_topk >= self.mpc_num_candidates:
                    raise ValueError(
                        "mpc_cem_topk must be < mpc_num_candidates."
                    )
                if self.mpc_cem_num_iter < 1:
                    raise ValueError("mpc_cem_num_iter must be ≥ 1.")
                if not (0.0 <= self.mpc_cem_anchor_blend <= 1.0):
                    raise ValueError("mpc_cem_anchor_blend must be in [0, 1].")
                if self.mpc_cem_init_mean not in ("anchor", "zero"):
                    raise ValueError(
                        f"mpc_cem_init_mean must be 'anchor' or 'zero'; got {self.mpc_cem_init_mean!r}"
                    )
                if self.mpc_cem_return not in ("best_ever", "final_mean"):
                    raise ValueError(
                        f"mpc_cem_return must be 'best_ever' or 'final_mean'; got {self.mpc_cem_return!r}"
                    )
            if self.mpc_scheme == "mppi":
                if self.mpc_mppi_temperature <= 0:
                    raise ValueError("mpc_mppi_temperature must be > 0.")
                if self.mpc_mppi_num_iter < 1:
                    raise ValueError("mpc_mppi_num_iter must be ≥ 1.")
            if self.mpc_score_floor_margin < 0:
                raise ValueError("mpc_score_floor_margin must be ≥ 0.")
            if self.mpc_icem_beta < 0:
                raise ValueError("mpc_icem_beta must be ≥ 0.")
            if self.mpc_horizon_mode not in ("single", "ar", "multi_offset"):
                raise ValueError(
                    f"mpc_horizon_mode must be 'single', 'ar', or 'multi_offset'; got {self.mpc_horizon_mode!r}"
                )
            if self.mpc_horizon_mode == "multi_offset":
                if len(self.mpc_offsets) != len(self.mpc_offset_weights):
                    raise ValueError(
                        f"mpc_offsets and mpc_offset_weights length mismatch: "
                        f"{len(self.mpc_offsets)} vs {len(self.mpc_offset_weights)}"
                    )
                if any(k < 1 for k in self.mpc_offsets):
                    raise ValueError("each mpc_offsets entry must be ≥ 1.")
                if any(w < 0 for w in self.mpc_offset_weights):
                    raise ValueError("each mpc_offset_weights entry must be ≥ 0.")
                if max(self.mpc_offsets) > self.chunk_size:
                    raise ValueError(
                        f"max(mpc_offsets)={max(self.mpc_offsets)} exceeds chunk_size={self.chunk_size}; "
                        "candidate chunks are not long enough."
                    )
                if self.mpc_action_k_max < max(self.mpc_offsets):
                    raise ValueError(
                        f"mpc_action_k_max={self.mpc_action_k_max} must be ≥ max(mpc_offsets)="
                        f"{max(self.mpc_offsets)} (per-slot action tensor must hold the longest offset)."
                    )
            if self.rtc_config is not None and self.rtc_config.enabled:
                raise ValueError("mpc_enabled and RTC are mutually exclusive.")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def latent_goal_offset(self) -> int:
        """Resolved LGE target offset: explicit knob if set, else chunk_size."""
        if self.latent_goal_target_offset is not None:
            return self.latent_goal_target_offset
        return self.chunk_size

    @property
    def observation_delta_indices(self) -> list:
        if self.latent_goal_enabled:
            return [0, self.latent_goal_offset]
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
