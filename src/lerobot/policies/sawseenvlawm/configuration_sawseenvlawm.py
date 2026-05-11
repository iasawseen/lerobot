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
    # Source of z_g going into the action expert during training:
    #   "encoded"  — frozen le-wm CLS of the dataset's chunk-end frame.
    #                Train-only; falls back to "predicted" at inference.
    #   "predicted" — Latent Goal Expert's clean prediction reconstructed from its
    #                velocity at the sampled flow-matching timestep
    #                (z_g_pred = latent_goal_x_t - t · v_latent_goal). Matches inference.
    latent_goal_inject_z_g_source: str = "encoded"
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
        if self.latent_goal_inject_z_g_source not in ("encoded", "predicted"):
            raise ValueError(
                f"latent_goal_inject_z_g_source must be 'encoded' or 'predicted'; "
                f"got {self.latent_goal_inject_z_g_source!r}"
            )
        if self.latent_goal_train_num_steps < 1:
            raise ValueError(
                "latent_goal_train_num_steps must be ≥ 1; got "
                f"{self.latent_goal_train_num_steps}"
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
