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


@PreTrainedConfig.register_subclass("sawseenvlaki")
@dataclass
class SawSeenVLAKIConfig(PreTrainedConfig):
    """SawSeenVLA + Knowledge Insulation (KI) with FAST action tokens.

    Structurally identical to SawSeenVLA (frozen SmolVLM2 backbone +
    flow-matching action expert) plus a second loss head that predicts
    discrete FAST action tokens autoregressively. The action expert
    consumes the VLM's K/V cache through a ``.detach()`` boundary, so
    the flow-matching gradient never updates VLM weights — the VLM only
    learns "what actions look like" through the FAST cross-entropy loss
    flowing back via LoRA adapters.

    See ``design/TODO.md`` Item 2 and ``design/SawSeenVLAKI.md``.
    """

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

    empty_cameras: int = 0
    adapt_to_pi_aloha: bool = False
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

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = False

    add_image_special_tokens: bool = False

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75

    min_period: float = 4e-3
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False
    compile_mode: str = "max-autotune"

    # ── Knowledge Insulation + FAST tokens ──────────────────────────
    # Off by default → behaves as a structurally-identical SawSeenVLA.
    # Flip to True to enable the KI training scheme: append FAST action
    # tokens to the language input, run a second CE loss head on them,
    # and detach VLM K/V before the action expert reads it.
    ki_enabled: bool = False
    # Loss weight λ in ``L = L_action + λ · L_ki_ce``. Starts at 1.0;
    # may need calibration if CE-in-nats and MSE-in-action-units² differ
    # too much in magnitude.
    ki_loss_weight: float = 1.0
    # Whether to detach the VLM K/V tensors going into the action
    # expert (the "I" in KI). True (default) is paper-faithful: the
    # flow-matching gradient never reaches the VLM, so VLM LoRA is
    # adapted by FAST CE only. False keeps CE as a pure auxiliary —
    # both FM and CE update VLM LoRA. Useful on LoRA-budget hardware
    # where the ~1 M-param LoRA subspace can't be split between two
    # objectives without the action expert paying a steep cost.
    ki_detach: bool = True

    # Path / repo id of the FAST action tokenizer (HuggingFace Hub).
    # The released ``lerobot/fast-action-tokenizer`` was trained on a
    # broad mixture; consider re-training a libero-specific one via
    # ``lerobot_train_tokenizer.py`` if KI underperforms.
    fast_action_tokenizer_path: str = "lerobot/fast-action-tokenizer"
    # Maximum number of FAST tokens per chunk. Variable-length output
    # from the tokenizer is right-padded to this length; the loss masks
    # the padded positions. Real LIBERO chunks (chunk_size=50,
    # action_dim=7) span 100-145 tokens depending on content; 160
    # covers the tail with margin and a truncation warning fires past
    # that. Bumping this directly grows the VLM prefix and
    # quadratically raises attention memory — re-tune ``batch_size``
    # if you change it.
    fast_max_action_tokens: int = 160
    # Size of the FAST action vocabulary. The released
    # ``lerobot/fast-action-tokenizer`` (``UniversalActionProcessor``)
    # exposes ``vocab_size=2048`` (256 reserved + 1792 BPE merges).
    # The fast head/embed table is sized to this to avoid out-of-range
    # token IDs.
    fast_vocab_size: int = 2048

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by sawseenvlaki for aloha real models. "
                "It is not ported yet in LeRobot."
            )
        # KI requires the action expert to use pure cross-attention to
        # the VLM (no joint self-attn layers). The joint self-attn path
        # entangles prefix↔suffix gradients within a single attention
        # call, making the K/V detach for KI poorly defined. Forcing
        # ``self_attn_every_n_layers=-1`` routes every layer through
        # ``forward_cross_attn_layer`` where the detach point is local.
        # Also matches the π0.5 / π0.6 recipe (pure cross-attn from
        # expert to VLM).
        if self.ki_enabled and self.self_attn_every_n_layers > 0:
            self.self_attn_every_n_layers = -1

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
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
