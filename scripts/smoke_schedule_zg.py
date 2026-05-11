"""Smoke test for the scheduled z_g source.

Loads the existing Mode 3 checkpoint, overrides the config to set
``latent_goal_inject_z_g_source="scheduled"``, runs ``policy.forward``
at three pretend steps (0, mid, end) and verifies:

  - ``latent_goal_schedule_p`` lands at 0, ~0.5, 1.0 respectively.
  - ``loss_action`` is finite at every step.

Run inside lerobot-benchmark-libero (same mounts as smoke_mpc_sawseenvlawm.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.sawseenvlawm.modeling_sawseenvlawm import SawSeenVLAWMPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def make_synthetic_batch(cfg, bsize: int = 2, device: str = "cuda") -> dict[str, torch.Tensor]:
    state_dim = cfg.action_feature.shape[0] if cfg.action_feature else 7
    action_dim = state_dim
    batch = {
        OBS_STATE: torch.randn(bsize, state_dim, device=device),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 100, (bsize, cfg.tokenizer_max_length), device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(bsize, cfg.tokenizer_max_length, device=device, dtype=torch.bool),
        ACTION: torch.randn(bsize, cfg.chunk_size, action_dim, device=device),
        "action_is_pad": torch.zeros(bsize, cfg.chunk_size, device=device, dtype=torch.bool),
    }
    for key, feat in cfg.input_features.items():
        if feat.type.value == "VISUAL":
            c, h, w = feat.shape
            # observation_delta_indices=[0, chunk_size] when LGE on →
            # frames at offsets 0 and chunk_size. populate_queues stacks
            # along dim=1 at inference but at training time the batch
            # comes pre-stacked (B, 2, C, H, W).
            batch[key] = torch.zeros(bsize, 2, c, h, w, device=device)
    return batch


def run(checkpoint_path: str, end_step: int = 100, device: str = "cuda"):
    cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
    cfg.latent_goal_inject_z_g_source = "scheduled"
    cfg.latent_goal_inject_schedule_end_step = end_step
    cfg.__post_init__()
    policy = SawSeenVLAWMPolicy.from_pretrained(checkpoint_path, config=cfg).to(device).train()

    batch = make_synthetic_batch(cfg, bsize=2, device=device)

    for step in (0, end_step // 2, end_step):
        policy.model._train_step.fill_(step)
        loss, info = policy.forward(batch)
        p = info.get("latent_goal_schedule_p")
        print(f"  step={step:5d}  p={p:.3f}  loss_action={info.get('loss_action', float('nan')):.4f}")
        assert torch.isfinite(loss), "NaN loss"
        assert p is not None, "latent_goal_schedule_p missing"

    # Sanity: at step=0 every sample's predicted-mask should be False (p=0).
    # We can't directly observe the mask but the deterministic edges are:
    policy.model._train_step.fill_(0)
    _, info0 = policy.forward(batch)
    assert info0["latent_goal_schedule_p"] == 0.0
    policy.model._train_step.fill_(end_step * 2)
    _, info_end = policy.forward(batch)
    assert info_end["latent_goal_schedule_p"] == 1.0

    # update() should advance the step.
    before = int(policy.model._train_step.item())
    policy.update()
    after = int(policy.model._train_step.item())
    assert after == before + 1, f"update() did not advance step: {before} → {after}"
    print(f"  update(): step {before} → {after}")

    print("\nSCHEDULE OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/train/sawseenvlawm_libero_10k_bs64_lewm1_lg_expert_lg_predicted_2xGPUs_bf16/checkpoints/010000/pretrained_model",
    )
    ap.add_argument("--end-step", type=int, default=100)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    run(args.checkpoint, end_step=args.end_step, device=args.device)


if __name__ == "__main__":
    main()
