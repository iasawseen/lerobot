from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("sawseenvla_qwen")
@dataclass
class SawSeenVLAQwenConfig(PreTrainedConfig):
    """SawSeenVLA-Qwen: Qwen3.5-0.8B as a black-box VL encoder, independent action expert
    that cross-attends to per-layer hidden states from the 6 full-attention anchor layers.

    Differs from SawSeenVLA (SmolVLM2) in that:
      - The VLM is NOT vivisected — we call Qwen as published and read its hidden_states.
      - The action expert is an INDEPENDENT transformer (not a clone of the VLM decoder).
      - Per-layer cross-attention uses re-projected Qwen hidden states (not shared K/V).
    """

    # Input / output structure
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

    # Padding for state/action vectors (matches SawSeenVLA convention).
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing: pin to a fixed resolution so the per-image visual-token count
    # is predictable. Qwen3-VL patch_size=16, spatial_merge_size=2 → 256/16/2 = 8×8 = 64
    # visual tokens per camera at 256×256.
    image_size: int = 256
    empty_cameras: int = 0

    # Tokenizer / language
    tokenizer_max_length: int = 48
    pad_language_to: str = "max_length"  # avoid recompile storm under torch.compile

    # Flow-matching decoding
    num_steps: int = 10
    min_period: float = 4e-3
    max_period: float = 4.0

    # VLM (frozen by default — only the expert + projections train)
    vlm_model_name: str = "Qwen/Qwen3.5-0.8B"
    load_vlm_weights: bool = True
    freeze_vlm: bool = True
    train_state_proj: bool = True

    # Per-layer cross-attention anchor layers — Qwen3.5-0.8B has full attention at
    # layers [3, 7, 11, 15, 19, 23] (the 6 GatedAttention layers in the
    # `[L,L,L,F]×6` hybrid pattern). We default to these as Scheme B anchors.
    # hidden_states[i] is the OUTPUT of layer i-1 (HF convention: index 0 = input
    # embeddings, indices 1..24 = layer 0..23 outputs), so anchors point at
    # layer-output indices and the expert sees 6 representation levels.
    qwen_cross_attn_layer_indices: tuple[int, ...] = (4, 8, 12, 16, 20, 24)

    # Action expert geometry
    expert_hidden_size: int = 512
    expert_num_heads: int = 8
    expert_num_kv_heads: int = 8  # no GQA in the expert by default
    expert_intermediate_size: int = 2048  # 4× hidden, SwiGLU
    expert_dropout: float = 0.0

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    compile_model: bool = False
    compile_mode: str = "max-autotune"

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"chunk_size={self.chunk_size} < n_action_steps={self.n_action_steps}"
            )
        if len(self.qwen_cross_attn_layer_indices) == 0:
            raise ValueError("qwen_cross_attn_layer_indices must be non-empty")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))

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
