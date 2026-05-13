#!/usr/bin/env python
"""Mine LIBERO trajectories with SawSeenVLA checkpoint(s) into a LeRobot v3 dataset.

The output matches the HuggingFaceVLA/libero schema 1:1 so it can be concatenated
with the expert dataset at conversion time (see le-wm/scripts/libero_to_h5.py):

    observation.images.image   (image, 256x256x3 uint8)   — agentview
    observation.images.image2  (image, 256x256x3 uint8)   — wrist eye-in-hand
    observation.state          (float32, (8,))            — [eef_pos(3), axis_angle(3), gripper_qpos(2)]
    action                     (float32, (7,))

A sidecar mining_log.json records per-episode (ckpt_step, suite, task_id, success,
num_frames, init_state_id) so we can ablate later without polluting the dataset
schema with a non-standard `ckpt_step` feature.

Run inside the lerobot-benchmark-libero Docker image with mounts for the policy
checkpoint root and the output dataset path. See `make -f sawseenvla.mk mine`.

Example:
    python scripts/mine_libero.py \\
      --ckpts /ckpts/002000/pretrained_model \\
              /ckpts/004000/pretrained_model \\
              /ckpts/006000/pretrained_model \\
      --eps-per-task 4 3 3 \\
      --suites libero_spatial libero_object libero_goal libero_10 \\
      --output-root /datasets/sawseenvla_libero_mined \\
      --repo-id local/sawseenvla_libero_mined
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins


LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
FPS = 10
IMG_H = IMG_W = 256

FEATURES: dict[str, dict] = {
    "observation.images.image":  {"dtype": "image",   "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "image",   "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channel"]},
    "observation.state":         {"dtype": "float32", "shape": (8,),              "names": ["state"]},
    "action":                    {"dtype": "float32", "shape": (7,),              "names": ["actions"]},
}


def icem_colored_noise(
    n_envs: int, T: int, action_dim: int,
    beta: float = 2.0, scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate per-env colored-noise action perturbations.

    Power spectrum density ∝ 1/f^beta along the time axis (Pinneri et al. 2020,
    "iCEM"). beta=0 = white Gaussian noise, beta=1 = pink, beta=2 = red/Brownian.
    Each env gets an independent sequence; per (env, dim) the std is normalized
    to 1.0 so the `scale` argument is directly the per-step std.

    Returns: (n_envs, T, action_dim) float32, ready to add to (action_chunk * scale).
    """
    rng = rng if rng is not None else np.random.default_rng()
    white = rng.standard_normal((n_envs, T, action_dim))
    freq = np.fft.rfft(white, axis=1)
    freqs = np.fft.rfftfreq(T)
    weights = np.zeros_like(freqs)
    if beta > 0 and len(freqs) > 1:
        weights[1:] = freqs[1:] ** (-beta / 2.0)
    else:  # white-noise fallback
        weights[1:] = 1.0
    colored = np.fft.irfft(freq * weights[None, :, None], n=T, axis=1)
    std = colored.std(axis=1, keepdims=True) + 1e-8
    colored = colored / std
    return (scale * colored).astype(np.float32)


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """numpy port of LiberoProcessorStep._quat2axisangle. Input (x, y, z, w)."""
    q = np.asarray(quat, dtype=np.float32).reshape(4)
    w = float(np.clip(q[3], -1.0, 1.0))
    den = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if den < 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * float(np.arccos(w))
    axis = q[:3] / den
    return (axis * angle).astype(np.float32)


def construct_state_per_env(obs_raw: dict, env_i: int) -> np.ndarray:
    """Build the 8-dim observation.state for env_i from the batched raw env obs.

    Matches HuggingFaceVLA/libero: [eef_pos(3), axis_angle(3), gripper_qpos(2)].
    """
    rs = obs_raw["robot_state"]
    eef_pos = np.asarray(rs["eef"]["pos"][env_i], dtype=np.float32).reshape(3)
    eef_quat = np.asarray(rs["eef"]["quat"][env_i], dtype=np.float32).reshape(4)
    grip = np.asarray(rs["gripper"]["qpos"][env_i], dtype=np.float32).reshape(2)
    axisangle = quat_to_axis_angle(eef_quat)
    return np.concatenate([eef_pos, axisangle, grip]).astype(np.float32)


def load_policy(checkpoint_path: Path, device: torch.device):
    """Dispatch to the concrete PreTrainedPolicy subclass via config.json:type."""
    cfg_path = checkpoint_path / "config.json"
    with cfg_path.open() as f:
        saved_cfg = json.load(f)
    policy_type = saved_cfg.get("type") or saved_cfg.get("policy_type")
    if not policy_type:
        raise ValueError(f"Couldn't find policy 'type' in {cfg_path}")
    PolicyCls = get_policy_class(policy_type)
    policy = PolicyCls.from_pretrained(str(checkpoint_path))
    policy = policy.to(device).eval()
    return policy, policy_type


def select_action_batched(
    policy,
    obs_raw: dict,
    *,
    task_descriptions: list[str],
    env_preprocessor,
    policy_preprocessor,
    policy_postprocessor,
    env_postprocessor,
    device: torch.device,
) -> np.ndarray:
    """Mirror lerobot_eval.rollout's policy call: preprocess raw obs, infer, postprocess."""
    obs = preprocess_observation(obs_raw)
    # Language-conditioned policies (SawSeenVLA) need the per-env task string.
    obs["task"] = list(task_descriptions)
    obs = env_preprocessor(obs)
    obs = policy_preprocessor(obs)
    with torch.inference_mode():
        action = policy.select_action(obs)
    action = policy_postprocessor(action)
    action_transition = env_postprocessor({ACTION: action})
    action = action_transition[ACTION]
    return action.to("cpu").numpy()  # (n_envs, action_dim)


def run_one_task(
    *,
    suite_name: str,
    task_id: int,
    policy,
    policy_pre,
    policy_post,
    env_pre,
    env_post,
    n_envs: int,
    device: torch.device,
    base_seed: int,
    action_noise_std: float = 0.0,
    noise_beta: float = 2.0,
    use_async_envs: bool = False,
):
    """Run one rollout of n_envs parallel envs for (suite_name, task_id).

    Returns a list of `n_envs` per-env episode dicts:
        {
            "frames": list of {observation.images.*, observation.state, action} dicts,
            "task": str,
            "init_state_id": int,
            "success": bool,
            "num_frames": int,
        }

    If action_noise_std > 0, iCEM-style colored noise (1/f^noise_beta) is added
    to the policy's action per step. Per-env noise sequences are independent.
    """
    env_cfg = LiberoEnvConfig(
        task=suite_name,
        task_ids=[task_id],
        obs_type="pixels_agent_pos",  # needed for robot_state → 8-dim state
        observation_height=IMG_H,
        observation_width=IMG_W,
    )
    envs_dict = env_cfg.create_envs(n_envs=n_envs, use_async_envs=use_async_envs)
    vec_env = envs_dict[suite_name][task_id]

    try:
        max_steps = vec_env.call("_max_episode_steps")[0]
        task_descriptions = list(vec_env.call("task_description"))
        # Clear the policy's per-rollout state (chunk buffer, RNN hidden, etc.)
        # between tasks — without this, the action queue from the previous task
        # leaks into the first few steps of this task.
        policy.reset()
        seeds = [base_seed + task_id * 1000 + i for i in range(n_envs)]
        obs_raw, info = vec_env.reset(seed=seeds)

        # Pre-generate the colored-noise trajectory for this task-run. Same
        # length as max_steps; indexed per env by global step counter (across
        # any in-task auto-resets — done envs stop recording but the index
        # advances anyway, which is fine since we never re-use those rows).
        noise = None
        if action_noise_std > 0.0:
            noise_rng = np.random.default_rng(base_seed + 42 * task_id)
            noise = icem_colored_noise(
                n_envs=n_envs, T=max_steps, action_dim=7,
                beta=noise_beta, scale=action_noise_std, rng=noise_rng,
            )

        # capture initial init_state_id from each sub-env (post-reset, so already bumped)
        try:
            init_ids = [int(s) - vec_env.num_envs for s in vec_env.call("init_state_id")]
        except (AttributeError, NotImplementedError):
            init_ids = [i for i in range(n_envs)]

        per_env_frames: list[list[dict]] = [[] for _ in range(n_envs)]
        per_env_success: list[bool] = [False] * n_envs
        done = np.zeros(n_envs, dtype=bool)

        for step in range(max_steps):
            if np.all(done):
                break

            # action chosen from current obs
            action_np = select_action_batched(
                policy,
                obs_raw,
                task_descriptions=task_descriptions,
                env_preprocessor=env_pre,
                policy_preprocessor=policy_pre,
                policy_postprocessor=policy_post,
                env_postprocessor=env_post,
                device=device,
            )  # (n_envs, 7)
            if noise is not None:
                action_np = np.clip(action_np + noise[:, step], -1.0, 1.0).astype(np.float32)

            # Record (obs, action) for the envs still running BEFORE stepping
            for i in range(n_envs):
                if done[i]:
                    continue
                img1 = np.asarray(obs_raw["pixels"]["image"][i], dtype=np.uint8)
                img2 = np.asarray(obs_raw["pixels"]["image2"][i], dtype=np.uint8)
                state = construct_state_per_env(obs_raw, i)
                per_env_frames[i].append({
                    "observation.images.image":  img1,
                    "observation.images.image2": img2,
                    "observation.state":         state,
                    "action":                    action_np[i].astype(np.float32),
                    "task":                      task_descriptions[i] if i < len(task_descriptions) else task_descriptions[0],
                })

            obs_raw, reward, terminated, truncated, info = vec_env.step(action_np)
            successes = info.get("is_success") if isinstance(info, dict) else None
            if successes is None and isinstance(info, dict) and "final_info" in info:
                fi = info["final_info"]
                if isinstance(fi, dict) and "is_success" in fi:
                    successes = fi["is_success"]
            if successes is not None:
                successes = np.asarray(successes).reshape(-1)
                for i in range(n_envs):
                    if i < successes.shape[0] and bool(successes[i]):
                        per_env_success[i] = True

            done = done | np.asarray(terminated).reshape(-1) | np.asarray(truncated).reshape(-1)
    finally:
        try:
            vec_env.close()
        except Exception:  # noqa: BLE001
            pass

    episodes = []
    for i in range(n_envs):
        episodes.append({
            "frames":        per_env_frames[i],
            "task":          task_descriptions[i] if i < len(task_descriptions) else task_descriptions[0],
            "init_state_id": init_ids[i] if i < len(init_ids) else i,
            "success":       per_env_success[i],
            "num_frames":    len(per_env_frames[i]),
        })
    return episodes


def main():
    p = argparse.ArgumentParser(
        description="Mine LIBERO trajectories with SawSeenVLA into a LeRobot v3 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="Paths to pretrained_model dirs (one per pass).")
    p.add_argument("--eps-per-task", nargs="+", type=int, required=True,
                   help="Episodes per (suite,task) for each --ckpt (same length).")
    p.add_argument("--output-root", type=Path, required=True,
                   help="Output dir for the LeRobot v3 dataset (created fresh).")
    p.add_argument("--repo-id", default="local/sawseenvla_libero_mined",
                   help="Repo id stored in the dataset metadata.")
    p.add_argument("--suites", nargs="+", default=LIBERO_SUITES,
                   help="LIBERO suite(s) to mine.")
    p.add_argument("--task-ids", nargs="+", type=int, default=None,
                   help="Optional subset of task_ids within each suite (default: all).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed for env init / per-env seeding.")
    p.add_argument("--action-noise-std", type=float, default=0.0,
                   help="iCEM-style colored-noise std added to policy actions per step "
                        "(0.0 = pure policy rollouts). Recommended: 0.05-0.1.")
    p.add_argument("--noise-beta", type=float, default=2.0,
                   help="Colored-noise power-spectrum exponent. 0=white, 1=pink, 2=red.")
    p.add_argument("--use-async-envs", action="store_true",
                   help="Use AsyncVectorEnv for per-env subprocess parallelism "
                        "(higher throughput at high n_envs, more startup cost).")
    args = p.parse_args()

    if len(args.ckpts) != len(args.eps_per_task):
        raise ValueError("--ckpts and --eps-per-task must have same length")

    register_third_party_plugins()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    args.output_root = args.output_root.resolve()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(
            f"output-root {args.output_root} exists and is non-empty; refusing to overwrite"
        )
    # LeRobotDataset.create() does mkdir(exist_ok=False); the parent must exist
    # but the leaf must not. Allow an existing empty leaf only if we can rmdir it
    # (same-UID); otherwise rely on the caller to pass a clean target.
    if args.output_root.exists():
        try:
            args.output_root.rmdir()
        except OSError as e:
            raise FileExistsError(
                f"output-root {args.output_root} exists but cannot be removed "
                f"(probably a docker bind-mount artifact): {e}. Bind-mount the "
                f"PARENT directory and pass --output-root /<parent_mount>/<leaf>."
            ) from e

    print(f"Creating dataset at {args.output_root}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=FEATURES,
        root=args.output_root,
        robot_type="panda",
        use_videos=False,
        image_writer_threads=4,
    )

    mining_log: list[dict] = []
    log_path = args.output_root / "mining_log.json"

    try:
        global_ep = 0
        for pass_idx, (ckpt_str, n_eps_per_task) in enumerate(zip(args.ckpts, args.eps_per_task)):
            ckpt_path = Path(ckpt_str)
            t_pass = time.time()
            print(f"\n=== Pass {pass_idx + 1}/{len(args.ckpts)} | ckpt={ckpt_path.name} | eps/task={n_eps_per_task} ===")

            policy, policy_type = load_policy(ckpt_path, device)
            print(f"  loaded policy_type={policy_type}")

            # NB: lerobot_eval passes `policy_cfg=cfg.policy`; we reuse the loaded
            # policy's own config since pretrained_path provides the processor stats.
            preprocessor_overrides = {
                "device_processor": {"device": str(policy.config.device)},
            }
            policy_pre, policy_post = make_pre_post_processors(
                policy_cfg=policy.config,
                pretrained_path=str(ckpt_path),
                preprocessor_overrides=preprocessor_overrides,
            )

            # env_cfg picks LiberoProcessorStep via libero registration. The
            # specific `task` value isn't used by get_env_processors().
            env_pre, env_post = make_env_pre_post_processors(
                env_cfg=LiberoEnvConfig(task=args.suites[0]),
                policy_cfg=policy.config,
            )

            for suite in args.suites:
                from libero.libero import benchmark
                suite_obj = benchmark.get_benchmark_dict()[suite]()
                total_tasks = len(suite_obj.tasks)
                task_ids = list(range(total_tasks)) if args.task_ids is None else args.task_ids
                for tid in task_ids:
                    t_task = time.time()
                    episodes = run_one_task(
                        suite_name=suite,
                        task_id=tid,
                        policy=policy,
                        policy_pre=policy_pre,
                        policy_post=policy_post,
                        env_pre=env_pre,
                        env_post=env_post,
                        n_envs=n_eps_per_task,
                        device=device,
                        base_seed=args.seed + pass_idx * 100_000,
                        action_noise_std=args.action_noise_std,
                        noise_beta=args.noise_beta,
                        use_async_envs=args.use_async_envs,
                    )

                    for ep in episodes:
                        if ep["num_frames"] == 0:
                            continue
                        for frame in ep["frames"]:
                            dataset.add_frame(frame)
                        dataset.save_episode()
                        mining_log.append({
                            "global_episode_index": global_ep,
                            # Most ckpt paths end in ".../<step>/pretrained_model"; use the
                            # parent dir as the short tag, fall back to .name otherwise.
                            "ckpt": ckpt_path.parent.name if ckpt_path.name == "pretrained_model" else ckpt_path.name,
                            "ckpt_path": str(ckpt_path),
                            "suite": suite,
                            "task_id": tid,
                            "task_description": ep["task"],
                            "init_state_id": ep["init_state_id"],
                            "success": ep["success"],
                            "num_frames": ep["num_frames"],
                        })
                        global_ep += 1

                    dt = time.time() - t_task
                    success_count = sum(1 for e in episodes if e["success"])
                    total_frames = sum(e["num_frames"] for e in episodes)
                    print(f"  {suite}/{tid:02d} | {len(episodes)} eps | {success_count}/{len(episodes)} success | "
                          f"{total_frames} frames | {dt:.1f}s")

                    # Flush mining log after each task so we don't lose state on crash.
                    with log_path.open("w") as f:
                        json.dump(mining_log, f, indent=2)

            del policy
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"=== Pass {pass_idx + 1} done in {time.time() - t_pass:.0f}s ===")
    finally:
        dataset.finalize()
        with log_path.open("w") as f:
            json.dump(mining_log, f, indent=2)

    print(f"\nMined {global_ep} episodes -> {args.output_root}")
    print(f"Mining log: {log_path}")


if __name__ == "__main__":
    main()
