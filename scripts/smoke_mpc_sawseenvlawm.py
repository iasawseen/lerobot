"""Smoke test for sawseenvlawm Phase B / MPC inference.

Loads a trained Mode 3 LGE checkpoint, overrides the config to enable
MPC, builds a synthetic 2-batch observation, and runs
``predict_action_chunk`` once each for ``mpc_scheme="anchor_perturb"``
and ``mpc_scheme="cem"``. Verifies output shape and absence of NaN/inf.

Run inside the lerobot-benchmark-libero image so torch + transformers
are available:

  docker run --gpus all --rm \
    -v ~/.cache/huggingface:/home/user_lerobot/.cache/huggingface \
    -v ~/.stable-wm/libero:/lewm:ro \
    -v $HOME/data/reps/le-wm:/lewm-src:ro \
    -v $(pwd)/outputs:/lerobot/outputs \
    -v $(pwd)/src:/lerobot/src \
    -v $(pwd)/scripts:/lerobot/scripts \
    -e PYTHONPATH=/lewm-src:/lerobot/src \
    -w /lerobot \
    lerobot-benchmark-libero \
    python scripts/smoke_mpc_sawseenvlawm.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.sawseenvlawm.configuration_sawseenvlawm import SawSeenVLAWMConfig
from lerobot.policies.sawseenvlawm.modeling_sawseenvlawm import SawSeenVLAWMPolicy
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def make_synthetic_batch(cfg: SawSeenVLAWMConfig, bsize: int = 2, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Build a minimal batch the policy can ingest end-to-end."""
    state_dim = cfg.action_feature.shape[0] if cfg.action_feature else 7
    # The policy expects normalized observations.
    batch = {
        OBS_STATE: torch.randn(bsize, state_dim, device=device),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 100, (bsize, cfg.tokenizer_max_length), device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(bsize, cfg.tokenizer_max_length, device=device, dtype=torch.bool),
    }
    # Images: one entry per camera, shape (B, 3, H, W) in [-1, 1].
    for key, feat in cfg.input_features.items():
        if feat.type.value == "VISUAL":
            c, h, w = feat.shape
            batch[key] = torch.zeros(bsize, c, h, w, device=device)
    return batch


def run(
    checkpoint_path: str,
    scheme: str,
    num_candidates: int = 4,
    bsize: int = 2,
    device: str = "cuda",
) -> torch.Tensor:
    """Load the policy with MPC overrides, run one predict_action_chunk."""
    print(f"\n=== Scheme={scheme!r}, N={num_candidates}, bs={bsize} ===")
    cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
    cfg.mpc_enabled = True
    cfg.mpc_scheme = scheme
    cfg.mpc_num_candidates = num_candidates
    cfg.mpc_noise_scale = 0.1
    cfg.mpc_cem_num_iter = 2
    cfg.mpc_cem_topk = min(2, num_candidates - 1)
    cfg.mpc_cem_anchor_blend = 0.5
    # Falls back to lewm_encoder_path (already set in the saved config).
    cfg.mpc_predictor_path = None
    cfg.__post_init__()  # re-run validations after mutation
    policy = SawSeenVLAWMPolicy.from_pretrained(checkpoint_path, config=cfg)
    policy = policy.to(device).eval()
    print(f"  policy loaded; mpc={policy.config.mpc_enabled}, "
          f"lewm_world={policy.model.lewm_world is not None}")

    batch = make_synthetic_batch(cfg, bsize=bsize, device=device)

    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)
    assert actions.shape[0] == bsize, f"unexpected batch dim: {actions.shape}"
    assert torch.isfinite(actions).all(), "NaN/inf in MPC output"
    action_dim = cfg.action_feature.shape[0]
    print(f"  output: shape={tuple(actions.shape)}, min={actions.min().item():.3f}, "
          f"max={actions.max().item():.3f}, action_dim={action_dim}")
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/train/sawseenvlawm_libero_10k_bs64_lewm1_lg_expert_lg_predicted_2xGPUs_bf16/checkpoints/010000/pretrained_model",
    )
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    for scheme in ("anchor_perturb", "cem"):
        run(args.checkpoint, scheme=scheme, num_candidates=args.n, bsize=args.bs, device=args.device)

    # Post-proj wiring sanity checks: load one more time and probe the
    # encoder's projector + BatchNorm modes directly.
    print("\n=== Post-projector wiring checks ===")
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.mpc_enabled = True
    cfg.mpc_scheme = "anchor_perturb"
    cfg.mpc_num_candidates = 4
    cfg.__post_init__()
    policy = SawSeenVLAWMPolicy.from_pretrained(args.checkpoint, config=cfg).to(args.device).eval()
    enc = policy.model.lewm_encoder

    assert enc.projector is not None, "lewm_encoder.projector not loaded"
    print(f"  projector loaded: {type(enc.projector).__name__}")

    img = torch.zeros(2, 3, 224, 448, device=args.device)
    raw = enc(img)[:, 0, :]
    proj = enc.encode_cls(img)
    delta = (raw - proj).abs().mean().item()
    print(f"  encode_cls vs raw CLS mean|Δ|: {delta:.4f} (projector is non-identity)")
    assert delta > 1e-3, "projector appears to be identity — load failure?"

    # BatchNorm mode should stay eval even after policy.train()
    policy.train()
    from torch import nn as _nn
    bn_modes = [m.training for m in enc.projector.modules() if isinstance(m, _nn.BatchNorm1d)]
    print(f"  projector BatchNorm.training after policy.train(): {bn_modes}")
    assert not any(bn_modes), "BatchNorm slipped into train mode — train() override missing"

    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
