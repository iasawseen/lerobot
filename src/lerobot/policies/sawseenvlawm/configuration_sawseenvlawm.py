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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("sawseenvlawm")
@dataclass
class SawSeenVLAWMConfig(PreTrainedConfig):
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

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by sawseenvlawm_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SawSeenVLAWM weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    # le-wm visual side-channel for the action expert. Vision tokens from a
    # frozen ViT-Tiny (le-wm Libero pretraining) are prepended to the action
    # expert's suffix, so the action expert attends to them directly without
    # routing through the SmolVLM prefix.
    # When None, this policy is structurally equivalent to vanilla
    # SawSeenVLA (but registered separately so checkpoints don't collide).
    lewm_encoder_path: str | None = None
    lewm_freeze: bool = True
    # Total tokens fed to the action expert, sliced from the ViT output:
    # last_hidden_state[:, :num_tokens] (CLS + first N-1 patches).
    lewm_num_tokens: int = 192
    # ViT input shape after concatenating cameras horizontally and resizing.
    # Defaults match le-wm's libero training: 256x512 raw → Resize(224)
    # preserves shorter side → 224x448. For non-libero embodiments adjust
    # ``lewm_image_width`` to ``num_cameras * lewm_image_height`` (or any
    # rectangular shape; HF ViT handles via interpolate_pos_encoding).
    lewm_image_height: int = 224
    lewm_image_width: int = 448
    lewm_patch_size: int = 14
    # Where the lewm tokens enter the model:
    #   "suffix" → projected to expert_hidden_size, prepended to the action
    #     expert's suffix.
    #   "none"   → encoder is loaded but no tokens flow into the action
    #     expert. The encoder is still available for other consumers (e.g.
    #     Future Sight target encoding); used to isolate Future Sight as the
    #     only le-wm pathway into the training signal.
    lewm_inject_to: str = "suffix"

    # Latent Goal Expert — implementation of the "Future Sight"
    # expert from design/future-sight-implicit-wm.md (Phase A — single-latent
    # implicit world modeling). When enabled, a second flow-matching expert
    # sits next to the action expert on the shared VLM backbone
    # (layer-interleaved per the twin-experts wrapper). It regresses to the
    # encoded latent of the frame at offset ``chunk_size`` from the anchor
    # observation — i.e. the observation that would follow the last action
    # of the chunk the action expert just emitted. Off by default;
    # structurally a no-op when False.
    latent_goal_enabled: bool = False
    # Loss weight λ in ``L = L_action + λ · L_latent_goal``.
    latent_goal_loss_weight: float = 1.0
    # Training signal for the Latent Goal Expert. ``"bc"`` = flow-matching MSE against the
    # encoded chunk-end frame. ``"contrastive"`` is reserved for a later
    # ablation (InfoNCE on goal-text vs. chunk-end-latent pairs).
    latent_goal_loss_type: str = "bc"
    # Flow-matching denoising steps for the Latent Goal Expert at inference (unused in
    # Phase A — there is no Latent Goal Expert inference path yet).
    latent_goal_num_steps: int = 10
    # Latent Goal Expert width relative to VLM hidden_size (mirrors
    # ``expert_width_multiplier`` for the action expert). 0.75 keeps it the
    # same width as the action expert so they're symmetric heads.
    latent_goal_expert_width_multiplier: float = 0.75
    # Number of Latent Goal Expert layers. -1 = match VLM depth (same default as
    # the action expert via ``num_expert_layers``).
    latent_goal_num_expert_layers: int = -1

    # ── Mode 3: Latent Goal Expert-conditioned action expert ──────────────────────
    # When True, the action expert's suffix is prepended with two tokens
    # [z_t, z_g] in le-wm space (projected to expert hidden). z_t is the
    # le-wm CLS of the current frame; z_g is either the encoded chunk-end
    # frame (source="encoded", training only) or Latent Goal Expert's predicted clean
    # latent reconstructed from its velocity (source="predicted", matches
    # inference). Training switches to a sequential 3-pass forward
    # (prefix → Latent Goal Expert → action) so the action expert can read Latent Goal Expert's output;
    # inference adds K Latent Goal Expert denoising steps before the action denoising
    # loop.
    latent_goal_inject_to_action: bool = False
    # Residual LGE: parameterize the LGE as predicting Δz_g = z_g - z_t
    # (a residual from the current frame to the chunk-end frame) instead
    # of the absolute z_g. The reconstruction is ẑ_g = z_t + Δẑ_g, used
    # for SIGReg and Mode-3 injection. Implications:
    #   * Identity at init: if the LGE output projector is zero-init,
    #     the model starts with ẑ_g = z_t (do-nothing baseline correct).
    #   * Smaller effective target norm: at small k_tail the world barely
    #     changes, so Δz_g is much smaller-norm than z_g. Easier target,
    #     stronger relative gradient.
    #   * Mode-3 injection: when this is on, the action expert sees a
    #     single bank of K Δẑ_g tokens projected via
    #     ``latent_goal_action_dz_proj`` (replacing the legacy
    #     ``[zt_emb, zg_emb]`` 2-token pair). The current observation is
    #     already in the suffix via the WM side-channel tokens, so
    #     re-injecting z_t through the LGE bank is redundant.
    #   * Multi-token-aware: when the JEPA emits ``(B, K, D)`` per state
    #     (Plan-B), the residual path keeps the K axis end-to-end (no
    #     ``reshape(B, K*D)`` collapse), and SIGReg follows le-wm's
    #     own pattern (``rearrange "b k d -> 1 (b k) d"``).
    latent_goal_residual: bool = False
    # Multi-k LGE training: when set to a comma-separated string of
    # integer offsets (e.g. ``"5,10,15,20,25"``), the LGE is trained
    # to predict residuals at multiple future horizons rather than the
    # single ``latent_goal_target_offset``. Training samples one k per
    # batch sample uniformly from the parsed offsets and conditions
    # the LGE on it via a new ``latent_goal_k_emb`` suffix token.
    # Dataset must return observations at all the offsets
    # (``observation_delta_indices`` = ``[0] + parsed_offsets``).
    # At inference, the LGE is queried at the single offset given by
    # ``latent_goal_target_offset`` — only the training signal is
    # multi-k. Empty string (default) = legacy single-k path.
    #
    # Note: a CSV string field, not a ``tuple[int, ...]``, because
    # draccus can't cleanly parse tuple literals from CLI strings.
    latent_goal_target_offsets: str = ""
    # Source of z_g going into the action expert during training:
    #   "encoded"  — frozen le-wm CLS of the dataset's chunk-end frame.
    #                Train-only; falls back to "predicted" at inference.
    #   "predicted" — Latent Goal Expert's clean prediction reconstructed from its
    #                velocity at the sampled flow-matching timestep
    #                (z_g_pred = latent_goal_x_t - t · v_latent_goal). Matches inference.
    #   "scheduled" — per-sample Bernoulli mix: at step s, each sample
    #                independently uses ``predicted`` with probability
    #                p = clamp(s / latent_goal_inject_schedule_end_step, 0, 1)
    #                and ``encoded`` otherwise. Ramps from 100% teacher
    #                (encoded) at step 0 → 100% student (predicted) at
    #                ``schedule_end_step``. Closes the train/eval z_g
    #                distribution gap gradually rather than at-once.
    latent_goal_inject_z_g_source: str = "encoded"
    # Step at which the ``scheduled`` source reaches 100% predicted
    # (linear ramp from ``latent_goal_inject_schedule_start_step``). Must
    # be > ``latent_goal_inject_schedule_start_step`` when
    # source="scheduled"; ignored otherwise. Typical value: equal to
    # ``scheduler_decay_steps`` so the schedule completes alongside the
    # LR cosine decay.
    latent_goal_inject_schedule_end_step: int = 0
    # Step at which the ``scheduled`` ramp *starts*. Before this step,
    # p=0 (pure ``encoded`` teacher). From start_step → end_step, p ramps
    # linearly from 0 → 1. Default 0 = ramp from the very beginning of
    # training (legacy behavior). Set > 0 to keep the action expert on
    # the clean teacher signal until LGE has had time to fit, then start
    # blending in its predictions. Ignored unless source="scheduled".
    latent_goal_inject_schedule_start_step: int = 0
    # Detach z_g (and z_t) before they enter the action expert. True is
    # the paper-faithful KI-style barrier — action loss cannot reshape
    # Latent Goal Expert weights through the conditioning path. False makes the
    # conditioning path differentiable so Latent Goal Expert also learns from action
    # loss (collapses goal latent toward "whatever helps the policy").
    latent_goal_inject_detach: bool = True
    # Number of LGE denoising steps used when reconstructing z_g for the
    # action expert during training (Mode 3, source="predicted"). 1
    # (default) = closed-form one-step reconstruction (z_g = x_t − t·v at
    # the single sampled t from the LGE training forward). K =
    # ``latent_goal_num_steps`` matches the eval inference loop — same
    # train/eval z_g distribution at the cost of K extra LGE forwards per
    # training step (~2–2.5× per-step wall time). The K extra forwards
    # run under ``torch.no_grad()``, so LGE training is unaffected: the
    # flow-matching loss still comes from the separate single-t LGE
    # forward. Ignored when source="encoded" or inject_to_action=False.
    latent_goal_train_num_steps: int = 1

    # ── SIGReg on the LGE's reconstructed clean prediction ───────────
    # Sketch Isotropic Gaussian Regularizer (Epps-Pulley statistic on
    # random 1-D projections; ported from le-wm/module.py:SIGReg). Pulls
    # the LGE's one-step-reconstructed clean prediction
    # ``z_g_pred = latent_goal_x_t - t · v`` toward an isotropic-Gaussian
    # marginal distribution. Off by default. Worth trying when Mode 3
    # source="predicted"/"scheduled" — the action expert is trained to
    # condition on z_g distributed like le-wm's projector output
    # (already SIGReg-shaped during JEPA training); SIGReg on z_g_pred
    # makes that distribution match explicit, in case MSE-against-target
    # isn't enough to land the marginals there on its own.
    # ``weight=0.0`` short-circuits the module entirely (no extra forward).
    # The signal is single-GPU; under DDP each rank computes the
    # statistic on its own batch shard.
    latent_goal_sigreg_weight: float = 0.0
    latent_goal_sigreg_knots: int = 17
    latent_goal_sigreg_num_proj: int = 1024

    # ── Phase B: MPC inference with le-wm predictor ──────────────────
    # Runtime-only addition. When True, ``sample_actions`` produces a
    # deterministic *anchor* action chunk first, then samples N
    # perturbations around it, rolls each through the le-wm predictor
    # in latent space, scores against the Latent Goal Expert's z_g,
    # and returns the argmin candidate. No training change — MPC is
    # never activated during ``forward`` (only via ``sample_actions``).
    mpc_enabled: bool = False
    # ``"anchor_perturb"`` = single-shot MPC: anchor + N-1 Gaussian
    # perturbations, pick argmin (anchor stays as candidate 0 so MPC
    # is no-worse than the bare policy). ``"cem"`` = CEM outer loop
    # over Gaussian perturbations centered on the anchor: M iters of
    # (sample, score, fit topk Gaussian, resample), return best.
    mpc_scheme: str = "anchor_perturb"
    # Total candidates per scoring batch. anchor_perturb includes the
    # anchor itself as candidate 0; cem uses this many candidates per
    # CEM iter.
    mpc_num_candidates: int = 16
    # Gaussian noise scale for action perturbations (post-normalization
    # action units; sawseenvlawm actions are unit-std-ish, so 0.1 ≈ 10%
    # of action std).
    mpc_noise_scale: float = 0.1
    # Scoring space is implicit: LeWM's projector is applied at the
    # encoder wrapper, and LGE is supervised in that same space, so
    # both z_t and z_g flow into MPC already post-projector. No knob.
    # CEM outer-loop iterations (Scheme B only).
    mpc_cem_num_iter: int = 4
    # CEM elite set size (Scheme B only). Must be < mpc_num_candidates.
    mpc_cem_topk: int = 4
    # σ anchoring weight toward mpc_noise_scale across CEM iters: new_σ =
    # elite.std() * (1 - blend) + mpc_noise_scale * blend. 1.0 = pure
    # init σ each iter; 0.0 = σ drifts freely from elites. Scheme B only.
    mpc_cem_anchor_blend: float = 0.5
    # ── CEM variant knobs (mirror le-wm's reference CEMSolver) ───────
    # Whether the policy anchor occupies candidate slot 0 every iter
    # (the AI-CEM variant). When False, slot 0 is the current μ instead
    # — matching le-wm's ``CEMSolver`` semantics (anchor never enters
    # the elite pool, so the search is unbiased by the policy prior).
    mpc_cem_include_anchor: bool = True
    # Initial μ_0. "anchor" = the policy's flow-matched chunk (default
    # — the policy prior seeds the search). "zero" = zeros, ignoring
    # the policy entirely (le-wm's default: ``init_action_distrib``
    # returns zeros when ``init_action=None``).
    mpc_cem_init_mean: str = "anchor"
    # What CEM returns at the end. "best_ever" tracks the
    # lowest-cost candidate seen across all iters (safer when M is
    # small and σ may not have converged). "final_mean" returns the
    # converged μ from the last iter (le-wm reference; relies on
    # σ-shrinkage to make μ a meaningful elite consensus).
    mpc_cem_return: str = "best_ever"
    # ── MPPI (Scheme C) — cost-weighted softmax aggregation ──────────
    # Temperature β for the softmax: weights_k = softmax(−cost_k / β).
    # β → 0  recovers hard argmin (every candidate's weight collapses
    #         onto the lowest-cost one).
    # β → ∞  recovers a uniform average over candidates.
    # Costs are sum-of-squares L2 in 192-d projector space, so the
    # absolute scale depends on the predictor and the scene; tune
    # against the cost distribution rather than against unit intuition.
    mpc_mppi_temperature: float = 1.0
    # MPPI outer-loop iterations. Default M=4 matches CEM's iteration
    # count and total predictor rollouts (N · M = 64), so MPPI-vs-CEM
    # is a fair head-to-head at fixed compute. M=1 recovers vanilla
    # single-shot MPPI — useful as a "softmax beats argmin at AP's
    # budget" sanity check, but under-explores in our one-shot-per-chunk
    # setup (no receding horizon to re-anchor μ).
    mpc_mppi_num_iter: int = 4
    # ── Score-floor escape (all schemes) ─────────────────────────────
    # Return the anchor unchanged unless the best candidate's cost is
    # materially lower: deviate only when
    #   (anchor_cost − best_cost) / anchor_cost ≥ mpc_score_floor_margin.
    # 0.0 (default) = disabled — return whatever the scheme picked, same
    # as the per-scheme best-ever-tracking floor (always ≤ anchor on
    # score, but vulnerable to predictor score-noise rank-inversion on
    # near-perfect anchors). 0.05 = require ≥ 5% relative improvement
    # before deviating. Higher values strengthen the anchor preference;
    # at margin → 1, MPC effectively becomes "return anchor" (no
    # candidate can clear a 100%+ improvement bar).
    # Per-batch (each chunk decision in the eval batch is gated
    # independently).
    mpc_score_floor_margin: float = 0.0
    # ── iCEM colored-noise exponent (all schemes) ────────────────────
    # Power-law spectral exponent for action-chunk perturbations: noise
    # PSD ∝ 1/f^β along the time axis. 0.0 (default) = white noise =
    # legacy behavior. 1.0 = pink. 2.0 = red/Brownian — the iCEM default
    # (Pinneri et al. 2020). Higher β = smoother, more temporally
    # correlated perturbations, which (1) stay closer to the manifold of
    # real action trajectories the le-wm predictor was trained on, and
    # (2) explore the cost surface in a structurally meaningful way
    # rather than as per-timestep jitter. Per-trajectory unit-std
    # normalization keeps the σ semantics from ``mpc_noise_scale``
    # unchanged. Works for cem and mppi (and anchor_perturb).
    mpc_icem_beta: float = 0.0
    # Path to a le-wm ``<name>_object.ckpt`` pickle that contains the
    # full JEPA (encoder + projector + action_encoder + predictor +
    # pred_proj). Falls back to ``lewm_encoder_path`` when None — the
    # same pickle works for both since it stores the whole module.
    mpc_predictor_path: str | None = None

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by sawseenvlawm for aloha real models. It is not ported yet in LeRobot."
            )
        if self.latent_goal_inject_to_action and not self.latent_goal_enabled:
            raise ValueError(
                "latent_goal_inject_to_action=True requires latent_goal_enabled=True — the "
                "action expert reads Latent Goal Expert-predicted z_g, so the Latent Goal Expert head must "
                "exist."
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
                "latent_goal_inject_z_g_source='scheduled' requires "
                "latent_goal_inject_schedule_end_step > 0 (the step at which "
                "the schedule reaches 100% predicted)."
            )
        if self.latent_goal_inject_schedule_start_step < 0:
            raise ValueError(
                "latent_goal_inject_schedule_start_step must be ≥ 0; got "
                f"{self.latent_goal_inject_schedule_start_step}"
            )
        if (
            self.latent_goal_inject_z_g_source == "scheduled"
            and self.latent_goal_inject_schedule_start_step
            >= self.latent_goal_inject_schedule_end_step
        ):
            raise ValueError(
                "latent_goal_inject_z_g_source='scheduled' requires "
                "latent_goal_inject_schedule_start_step < "
                "latent_goal_inject_schedule_end_step; got start="
                f"{self.latent_goal_inject_schedule_start_step} end="
                f"{self.latent_goal_inject_schedule_end_step}"
            )
        if self.latent_goal_train_num_steps < 1:
            raise ValueError(
                "latent_goal_train_num_steps must be ≥ 1; got "
                f"{self.latent_goal_train_num_steps}"
            )
        if self.latent_goal_sigreg_weight < 0:
            raise ValueError(
                "latent_goal_sigreg_weight must be ≥ 0; got "
                f"{self.latent_goal_sigreg_weight}"
            )
        if self.latent_goal_sigreg_weight > 0 and not self.latent_goal_enabled:
            raise ValueError(
                "latent_goal_sigreg_weight > 0 requires latent_goal_enabled=True — "
                "SIGReg targets the LGE's reconstructed z_g_pred, which only exists "
                "when the Latent Goal Expert is on."
            )
        if self.latent_goal_sigreg_knots < 2:
            raise ValueError(
                f"latent_goal_sigreg_knots must be ≥ 2; got {self.latent_goal_sigreg_knots}"
            )
        if self.latent_goal_sigreg_num_proj < 1:
            raise ValueError(
                f"latent_goal_sigreg_num_proj must be ≥ 1; got {self.latent_goal_sigreg_num_proj}"
            )
        if self.mpc_enabled:
            if not self.latent_goal_enabled:
                raise ValueError(
                    "mpc_enabled=True requires latent_goal_enabled=True — the Latent "
                    "Goal Expert is the source of z_g that MPC scores candidates against."
                )
            if not self.lewm_encoder_path and not self.mpc_predictor_path:
                raise ValueError(
                    "mpc_enabled=True requires a le-wm checkpoint (set "
                    "lewm_encoder_path or mpc_predictor_path)."
                )
            if self.mpc_num_candidates < 2:
                raise ValueError(
                    "mpc_num_candidates must be ≥ 2 (anchor + at least one "
                    f"perturbation); got {self.mpc_num_candidates}"
                )
            if self.mpc_scheme not in ("anchor_perturb", "cem", "mppi"):
                raise ValueError(
                    f"mpc_scheme must be 'anchor_perturb', 'cem', or 'mppi'; "
                    f"got {self.mpc_scheme!r}"
                )
            if self.mpc_scheme == "cem":
                if self.mpc_cem_topk >= self.mpc_num_candidates:
                    raise ValueError(
                        "mpc_cem_topk must be < mpc_num_candidates; got "
                        f"topk={self.mpc_cem_topk}, num_candidates={self.mpc_num_candidates}"
                    )
                if self.mpc_cem_num_iter < 1:
                    raise ValueError(
                        f"mpc_cem_num_iter must be ≥ 1; got {self.mpc_cem_num_iter}"
                    )
                if not (0.0 <= self.mpc_cem_anchor_blend <= 1.0):
                    raise ValueError(
                        "mpc_cem_anchor_blend must be in [0, 1]; got "
                        f"{self.mpc_cem_anchor_blend}"
                    )
                if self.mpc_cem_init_mean not in ("anchor", "zero"):
                    raise ValueError(
                        "mpc_cem_init_mean must be 'anchor' or 'zero'; got "
                        f"{self.mpc_cem_init_mean!r}"
                    )
                if self.mpc_cem_return not in ("best_ever", "final_mean"):
                    raise ValueError(
                        "mpc_cem_return must be 'best_ever' or 'final_mean'; got "
                        f"{self.mpc_cem_return!r}"
                    )
            if self.mpc_scheme == "mppi":
                if self.mpc_mppi_temperature <= 0:
                    raise ValueError(
                        "mpc_mppi_temperature must be > 0; got "
                        f"{self.mpc_mppi_temperature}"
                    )
                if self.mpc_mppi_num_iter < 1:
                    raise ValueError(
                        f"mpc_mppi_num_iter must be ≥ 1; got {self.mpc_mppi_num_iter}"
                    )
            if self.mpc_score_floor_margin < 0:
                raise ValueError(
                    "mpc_score_floor_margin must be ≥ 0; got "
                    f"{self.mpc_score_floor_margin}"
                )
            if self.mpc_icem_beta < 0:
                raise ValueError(
                    "mpc_icem_beta must be ≥ 0 (0=white, 1=pink, 2=red); got "
                    f"{self.mpc_icem_beta}"
                )
            if self.rtc_config is not None and self.rtc_config.enabled:
                raise ValueError(
                    "mpc_enabled and RTC are mutually exclusive — RTC partially "
                    "re-denoises chunks mid-execution, while MPC filters the full "
                    "chunk; combining them is not specified."
                )

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
    def parsed_latent_goal_target_offsets(self) -> list[int]:
        """Parse the CSV ``latent_goal_target_offsets`` field.

        Returns a sorted, deduped list of positive integers (e.g.
        ``[5, 10, 15, 20, 25]``). Empty string yields an empty list
        (= single-k legacy path is engaged elsewhere).
        """
        s = (self.latent_goal_target_offsets or "").strip()
        if not s:
            return []
        try:
            vals = [int(x) for x in s.split(",") if x.strip()]
        except ValueError as e:
            raise ValueError(
                f"latent_goal_target_offsets must be comma-separated ints, "
                f"got {self.latent_goal_target_offsets!r}"
            ) from e
        if any(v <= 0 for v in vals):
            raise ValueError(
                f"latent_goal_target_offsets must be positive, got {vals}"
            )
        return sorted(set(vals))

    @property
    def observation_delta_indices(self) -> list:
        # When the Latent Goal Expert is enabled, also fetch the frame at ``chunk_size``
        # offset from the anchor — that's the Latent Goal Expert regression target. The
        # dataset returns it alongside the anchor frame, padded if it falls
        # past the episode end (the Latent Goal Expert loss masks padded samples).
        if self.latent_goal_enabled:
            return [0, self.chunk_size]
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
