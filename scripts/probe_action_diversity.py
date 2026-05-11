"""Probe how much the action expert's output varies with the noise seed.

For a fixed observation, run sample_actions with K different noise samples and
report:
  * per-dim std across the K samples (averaged over chunk timesteps)
  * per-timestep std (averaged over dims)
  * mean pairwise L2 distance between sampled chunks
  * dataset-wide action std (the ceiling — if every state were equally likely
    in the dataset, the policy's per-obs variance can at most match this)

A useful read:
  * per-noise std ≪ dataset std → policy collapses to a deterministic action
    given the obs (good if the dataset is unimodal; bad if multimodal)
  * per-noise std ≈ dataset std → policy hasn't learned obs→action mapping
    (effectively random)
  * something in between → some learned multimodality / residual stochasticity

Run inside the lerobot-benchmark-libero Docker image, e.g.:
    docker run --gpus all --rm \\
      -v ~/.cache/huggingface:/home/user_lerobot/.cache/huggingface \\
      -v $(pwd)/outputs:/lerobot/outputs \\
      -v $(pwd)/src:/lerobot/src \\
      -v $(pwd)/scripts:/lerobot/scripts \\
      -e PYTHONPATH=/lerobot/src \\
      -w /lerobot \\
      lerobot-benchmark-libero \\
      python scripts/probe_action_diversity.py \\
        --policy outputs/train/sawseenvla_libero_96k_bs64_2xGPUs_bf16/checkpoints/last/pretrained_model
"""

from __future__ import annotations

import argparse
import time

import torch

import json
import os

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="Path to saved pretrained_model dir")
    p.add_argument("--dataset", default="HuggingFaceVLA/libero")
    p.add_argument("--num-obs", type=int, default=4, help="Distinct observations to probe")
    p.add_argument("--num-noise", type=int, default=32, help="Noise samples per observation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"Loading policy from {args.policy} …")
    t0 = time.time()
    # Read the saved config to find the policy type, then dispatch to the
    # concrete subclass — PreTrainedPolicy is abstract.
    with open(os.path.join(args.policy, "config.json")) as f:
        saved_cfg = json.load(f)
    policy_type = saved_cfg.get("type") or saved_cfg.get("policy_type")
    if not policy_type:
        raise ValueError(f"Couldn't find 'type' in {args.policy}/config.json")
    PolicyCls = get_policy_class(policy_type)
    policy = PolicyCls.from_pretrained(args.policy)
    policy = policy.to(device).eval()
    print(f"  policy_type={policy_type}, class={PolicyCls.__name__}")
    print(f"  loaded in {time.time() - t0:.1f}s; chunk_size={policy.config.chunk_size}, "
          f"max_action_dim={policy.config.max_action_dim}, "
          f"num_steps={policy.config.num_steps}")

    print(f"\nLoading dataset {args.dataset} …")
    t0 = time.time()
    ds = LeRobotDataset(args.dataset, episodes=[0])
    print(f"  ready in {time.time() - t0:.1f}s; len={len(ds)}")

    print(f"\nSampling {args.num_obs} observations …")
    obs_indices = torch.randperm(len(ds))[: args.num_obs].tolist()
    K, T, A_pad = args.num_noise, policy.config.chunk_size, policy.config.max_action_dim
    A = ds.features[ACTION]["shape"][0]

    pre, post = make_pre_post_processors(policy.config, dataset_stats=ds.meta.stats)

    print(f"  K={K} noise samples per obs; chunk={T}; action_dim={A}")

    # Dataset action stats (for the ceiling baseline)
    a_mean = ds.meta.stats[ACTION]["mean"]
    a_std = ds.meta.stats[ACTION]["std"]
    if not torch.is_tensor(a_mean):
        a_mean = torch.tensor(a_mean)
    if not torch.is_tensor(a_std):
        a_std = torch.tensor(a_std)
    print(f"\nDataset action std (raw, the upper-bound ceiling):")
    print("  " + ", ".join(f"{x:.3f}" for x in a_std.cpu().tolist()))

    print("\n=== per-observation probe ===")
    all_per_dim_stds = []
    all_pairwise_l2 = []
    all_means = []

    for obs_i, idx in enumerate(obs_indices):
        sample = ds[idx]
        # Apply the pre-processor to get a model-ready batch (single sample).
        single = pre(sample)
        # Replicate to K copies for batched sample_actions
        batch_k = {}
        for k, v in single.items():
            if torch.is_tensor(v):
                batch_k[k] = v.expand(K, *v.shape[1:]).contiguous()
            else:
                batch_k[k] = v

        noise = torch.randn(K, T, A_pad, device=device)
        t0 = time.time()
        with torch.no_grad():
            actions_norm = policy.predict_action_chunk(batch_k, noise=noise)
        dt = time.time() - t0
        # actions: (K, T, A) in the model's *normalized* (post action_out_proj
        # → pre-unnormalize) space, sliced to the dataset action_dim.

        # Diversity stats in normalized space
        per_dim_std = actions_norm.std(dim=0).mean(dim=0)  # (A,)
        per_time_std = actions_norm.std(dim=0).mean(dim=1)  # (T,)
        total_std = actions_norm.std(dim=0).mean().item()
        per_step_mean_action = actions_norm.mean(dim=0)  # (T, A)

        # Mean pairwise L2 (over the chunk × dim)
        diffs = actions_norm[:, None] - actions_norm[None, :]
        pl2 = diffs.flatten(2).norm(dim=-1)  # (K, K)
        off_diag = pl2[~torch.eye(K, dtype=torch.bool, device=pl2.device)]
        mean_pairwise = off_diag.mean().item()

        # Std of dataset normalized actions over the chunk_size window — this is
        # ~1 by construction (MEAN_STD norm). We compare per_dim_std against this.
        # In normalized space, the dataset std is exactly 1 per dim.

        print(f"\nobs[{obs_i}]  ds_idx={idx}  ({K} noise samples; inference {dt:.2f}s)")
        print(f"  per-dim std (normalized): " + ", ".join(f"{x:.3f}" for x in per_dim_std.cpu().tolist()))
        print(f"  mean per-dim std:         {per_dim_std.mean().item():.3f}    (1.0 = full dataset spread, 0 = collapsed)")
        print(f"  total std (mean over T,A):{total_std:.3f}")
        print(f"  mean pairwise L2:         {mean_pairwise:.3f}    (chunk*dim sqrt-sum)")
        print(f"  per-time std summary:     mean={per_time_std.mean().item():.3f}  "
              f"min={per_time_std.min().item():.3f}  max={per_time_std.max().item():.3f}")

        all_per_dim_stds.append(per_dim_std.cpu())
        all_pairwise_l2.append(mean_pairwise)
        all_means.append(per_step_mean_action.cpu())

    print("\n=== sanity: same noise → same action (must be ~0) ===")
    sample = ds[obs_indices[0]]
    single = pre(sample)
    batch_2 = {k: (v.expand(2, *v.shape[1:]).contiguous() if torch.is_tensor(v) else v) for k, v in single.items()}
    fixed_noise = torch.randn(1, T, A_pad, device=device).expand(2, -1, -1).contiguous()
    with torch.no_grad():
        a = policy.predict_action_chunk(batch_2, noise=fixed_noise)
    delta = (a[0] - a[1]).abs().max().item()
    print(f"  max abs diff between two runs with identical noise: {delta:.6e}  (expect ~0)")

    print("\n=== summary across observations ===")
    avg_per_dim_std = torch.stack(all_per_dim_stds).mean(dim=0)
    print(f"  avg per-dim std across {len(obs_indices)} obs:")
    print("  " + ", ".join(f"{x:.3f}" for x in avg_per_dim_std.tolist()))
    print(f"  mean: {avg_per_dim_std.mean().item():.3f}    (1.0 = dataset-wide spread)")
    print(f"  avg mean pairwise L2: {sum(all_pairwise_l2) / len(all_pairwise_l2):.3f}")


if __name__ == "__main__":
    main()
