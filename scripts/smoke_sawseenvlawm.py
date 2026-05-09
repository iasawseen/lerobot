"""Forward + sample_actions smoke test for SawSeenVLAWM.

Run inside the lerobot-benchmark-libero Docker image with le-wm sources on
PYTHONPATH and the lewm checkpoint mounted at /lewm/<ckpt>:

    docker run --gpus all --rm \\
      -v ~/.stable-wm/libero:/lewm:ro \\
      -v ~/data/reps/le-wm:/lewm-src:ro \\
      -v $(pwd)/src:/lerobot/src \\
      -e PYTHONPATH=/lewm-src:/lerobot/src \\
      -w /lerobot \\
      lerobot-benchmark-libero \\
      python scripts/smoke_sawseenvlawm.py
"""

from __future__ import annotations

import math
import os
import time

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.sawseenvlawm.configuration_sawseenvlawm import SawSeenVLAWMConfig
from lerobot.policies.sawseenvlawm.modeling_sawseenvlawm import SawSeenVLAWMPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def make_libero_like_config(lewm_path: str | None) -> SawSeenVLAWMConfig:
    state_dim = 8
    action_dim = 7
    img_h, img_w = 256, 256
    cfg = SawSeenVLAWMConfig(
        chunk_size=10,
        n_action_steps=10,
        max_state_dim=32,
        max_action_dim=32,
        load_vlm_weights=False,
        compile_model=False,
        pad_language_to="max_length",
        tokenizer_max_length=16,
        num_steps=2,
        num_vlm_layers=4,
        self_attn_every_n_layers=2,
        lewm_encoder_path=lewm_path,
        lewm_freeze=True,
        lewm_num_tokens=192,
    )
    cfg.input_features = {
        f"{OBS_IMAGES}.image_0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, img_h, img_w)),
        f"{OBS_IMAGES}.image_1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, img_h, img_w)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    cfg.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    cfg.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }
    return cfg


def make_dummy_batch(cfg: SawSeenVLAWMConfig, device: torch.device, batch_size: int = 2):
    img_keys = [k for k in cfg.input_features if k.startswith(OBS_IMAGES)]
    state_dim_dataset = cfg.input_features[OBS_STATE].shape[0]
    action_dim_dataset = cfg.output_features[ACTION].shape[0]
    img_h, img_w = cfg.input_features[img_keys[0]].shape[1:]

    batch = {}
    for k in img_keys:
        batch[k] = torch.rand(batch_size, 3, img_h, img_w, device=device) * 2 - 1  # [-1, 1]
    batch[OBS_STATE] = torch.randn(batch_size, state_dim_dataset, device=device)
    batch[ACTION] = torch.randn(batch_size, cfg.chunk_size, action_dim_dataset, device=device)

    seq_len = cfg.tokenizer_max_length
    batch[OBS_LANGUAGE_TOKENS] = torch.randint(0, 32_000, (batch_size, seq_len), device=device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    return batch


def main():
    lewm_path = os.environ.get("LEWM_CKPT", "/lewm/lewm_epoch_10_object.ckpt")
    if not os.path.exists(lewm_path):
        raise FileNotFoundError(f"lewm checkpoint not found: {lewm_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== SawSeenVLAWM smoke test ===")
    print(f"Device: {device}")
    print(f"lewm checkpoint: {lewm_path}")

    print("\n[1/3] Building config + policy (lewm enabled)...")
    cfg = make_libero_like_config(lewm_path=lewm_path)
    t0 = time.time()
    policy = SawSeenVLAWMPolicy(cfg).to(device).eval()
    print(f"   policy built in {time.time() - t0:.1f}s")

    enc = policy.model.lewm_encoder
    proj = policy.model.lewm_proj
    assert enc is not None, "lewm_encoder is None — config didn't take effect"
    assert proj is not None, "lewm_proj is None"
    print(
        f"   encoder = ViT(hidden={enc.output_dim}, num_tokens={enc.num_tokens}, "
        f"frozen={enc.freeze}); proj: {proj.in_features}->{proj.out_features}"
    )

    print("\n[2/3] Forward pass on dummy batch...")
    batch = make_dummy_batch(cfg, device, batch_size=2)
    t0 = time.time()
    loss, loss_dict = policy.forward(batch)
    print(f"   forward in {time.time() - t0:.1f}s; loss={loss.item():.4f}")
    assert math.isfinite(loss.item()), f"non-finite loss: {loss}"

    print("\n[3/3] Inference (predict_action_chunk)...")
    t0 = time.time()
    actions = policy.predict_action_chunk(batch)
    print(f"   inference in {time.time() - t0:.1f}s; actions.shape={tuple(actions.shape)}")
    assert actions.shape == (2, cfg.chunk_size, cfg.output_features[ACTION].shape[0])
    assert torch.isfinite(actions).all(), "non-finite values in predicted actions"

    n_params = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\nParam counts: total={n_params / 1e6:.1f}M, trainable={n_train / 1e6:.1f}M")
    print("\nOK: SawSeenVLAWM forward + sample_actions both work.")


if __name__ == "__main__":
    main()
