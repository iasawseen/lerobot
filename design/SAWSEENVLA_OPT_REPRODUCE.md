# SawSeenVLA Inference Optimization — Reproduction Runbook (RTX 3090 + Jetson Orin)

*Self-contained, copy-pasteable guide for another agent to reproduce the inference
optimization of SawSeenVLA and its benchmarks on **RTX 3090 Ti** and **Jetson AGX
Orin 64 GB**. Companion to the analysis doc `design/SAWSEENVLA_INFERENCE_OPT.md`
(the "why"); this file is the "how". All numbers here are measured, not estimated.*

> **What is SawSeenVLA?** A structural clone of **SmolVLA** — same frozen SmolVLM2-500M
> backbone (truncated to 16 of 32 layers, PixelShuffle → 64 vision tokens/frame) + a
> flow-matching action expert (K-step Euler denoise), chunk_size=50, bf16, ~450M params.
> Registered as `--policy.type=sawseenvla`. The optimization below applies unchanged to
> SmolVLA itself (the graphed path lives partly in the shared `smolvla/` module).

---

## 0. TL;DR — the result

The optimization is **two zero-GPU-h levers**, no distillation / quantization / retrain:

1. **K=10 → K=1** denoise steps (`--policy.num_steps=1`). Free here — the flow head
   converges in one Euler step; SR is unchanged (actually +pp).
2. **Two-graph CUDA capture** (`SAWSEEN_CUDAGRAPH=1`) — collapses the per-step kernel-launch
   overhead at bs=1. **Bit-exact** to eager (max |Δ| = 0.000000).

| config | RTX 3090 Ti | AGX Orin (MAXN) | SR (libero_spatial, 100 ep) |
|---|---:|---:|---|
| K=10 eager (baseline) | 218 ms | 734.3 ms | 75% |
| K=1 eager | 55.7 ms | 180.2 ms | 77% |
| **K=1 two-graph (optimized)** | **25.4 ms** | **66.8 ms** | **79%** |
| speedup vs baseline | **8.6×** | **11.0×** | no SR loss |

Orin 11.0× decomposes as **4.07× (K=10→K=1) × 2.70× (CUDA-graph)**. Bit-exactness holds
on sm_87 exactly as on the 3090. Commits: `ca2de108` (optimization), `c716390a` (Orin
benchmark + `Dockerfile.orin`), `d125a776` (Orin LIBERO-eval image).

---

## 1. The optimization (code) — commit `ca2de108`

Three changes, all behind env gates (default off → other policies/training unchanged).

### 1.1 `src/lerobot/policies/sawseenvla/modeling_sawseenvla.py`
- **`embed_suffix` host-sync fix.** Replaced `att_masks = torch.tensor(att_masks, …)`
  (a per-denoise-step host→device sync that blocks CUDA-graph capture) with a native
  `torch.ones(n_att, …)`. This is unconditional (it's a strict improvement).
- **`sample_actions` gate** (~line 852): when `os.environ["SAWSEEN_CUDAGRAPH"]=="1"` and
  RTC is disabled, dispatch to `_sample_actions_graphed(...)`.
- **`_sample_actions_graphed` + `_capture_inference_graphs`** (~lines 946/986): the
  **two-graph design**. The SmolVLM vision encoder stays eager (HF image preprocessing
  isn't capturable); everything after is captured as **two chained CUDA graphs sharing a
  persistent static KV-cache**:
  - *graph 1* — the VLM prefix forward (`fill_kv_cache=True`) recomputes K/V and copies
    them **in-place** into a once-allocated `static_cache`;
  - *graph 2* — the K-step flow denoise loop, reading that `static_cache`.
  Per call: copy fresh prefix embeddings/masks/positions into graph 1's static input
  buffers, replay graph 1, then loop-replay graph 2 with the Euler update
  `x_t += dt·v_t` in Python between replays. Re-captures on shape (batch/seq/K) change.

  **Why two graphs, not one:** a single combined capture replays *stale* K/V for new
  inputs → wrong actions (measured 26% SR). Splitting so graph 1 recomputes the cache
  and graph 2 reads it makes it bit-exact. Validate with a **2-input diagnostic**
  (feed input A then input B; assert graphed==eager for both and A≠B).

### 1.2 `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
- **`sdpa_attention_forward`** + gate (~line 532): `SAWSEEN_ATTN=sdpa` swaps the eager
  fp32-upcast attention for fused SDPA. **~0 gain at bs=1** (the regime is launch-overhead-
  bound, not matmul-bound) — kept as an opt-in for the batched/compute-bound regime.

### 1.3 Knobs summary
| knob | how | effect |
|---|---|---|
| denoise steps K | `--policy.num_steps=N` (config default 10) | K=1 is the free latency lever |
| CUDA-graph path | env `SAWSEEN_CUDAGRAPH=1` | the 2.2–2.7× launch-overhead win, bit-exact |
| fused attention | env `SAWSEEN_ATTN=sdpa` | batched only; ~0 at bs=1 |

---

## 2. Reproduce on RTX 3090 / 3090 Ti

Prereqs: the repo, the `lerobot-benchmark-libero` docker image (`make -f sawseenvla.mk build`),
a trained checkpoint (here `outputs/train/sawseenvla_libero_spatial_4k_bs96_1xGPU_full_bf16/checkpoints/last/pretrained_model`).

### 2.1 Latency (single-process, eager vs two-graph, bit-exact check)
`docker/bench_orin.py` is platform-agnostic (it just needs torch+CUDA+lerobot). On the
3090 run it in the benchmark image; it prints the 4-config table and the bit-exact diff:
```bash
docker run --rm --gpus all -e HF_HUB_OFFLINE=0 \
  -v $(pwd)/src:/lerobot/src -v $(pwd)/outputs:/lerobot/outputs \
  -v $(pwd)/docker/bench_orin.py:/lerobot/bench_orin.py -w /lerobot \
  lerobot-benchmark-libero python3 bench_orin.py \
  /lerobot/outputs/train/sawseenvla_libero_spatial_4k_bs96_1xGPU_full_bf16/checkpoints/last/pretrained_model
```
Expected (3090 Ti): K=10 eager 218 ms · K=1 eager 55.7 ms · K=1 two-graph 25.4 ms · diff 0.000000.

### 2.2 SR (the real harness, 100-episode protocol)
The number that certifies "no SR loss". `--eval.n_episodes=10` on `libero_spatial`
expands to 100 (10 tasks × 10 ep):
```bash
make -f sawseenvla.mk eval \
  EVAL_POLICY=outputs/train/sawseenvla_libero_spatial_4k_bs96_1xGPU_full_bf16/checkpoints/last/pretrained_model \
  EVAL_TASKS=libero_spatial EVAL_EPISODES=10 EVAL_BATCH=10
# optimized path: add SAWSEEN_CUDAGRAPH=1 to the docker env and --policy.num_steps=1
```
Or directly: `... -e SAWSEEN_CUDAGRAPH=1 ... lerobot-eval --policy.num_steps=1 --env.type=libero --env.task=libero_spatial --eval.n_episodes=10 --eval.batch_size=10 ...`.
Expected: pc_success ≈ 79 (vs 75 at K=10) → no regression.

---

## 3. Reproduce on Jetson AGX Orin 64 GB

**Target:** JetPack 6.2 (L4T R36.4.x), aarch64, Tegra iGPU (sm_87), nvidia = default docker
runtime. **Key environment facts that shaped everything:**
- **Root eMMC is small/full** → stage everything on the **NVMe at `/mnt/nvme`** (docker
  data-root is already there). Use `/mnt/nvme/lerobot-bench/` as the staging root.
- The uplink is **slow and lossy** → all transfers use `rsync --partial`; the libero
  build uses a BuildKit pip cache mount + retry loop (see §3.5).
- The device **rebooted once under load** mid-session — NVMe staging survives reboot;
  re-check reachability if it drops.

### 3.0 Staging root
```bash
ssh nvidia@<orin>           # this device: nvidia@10.10.0.34
mkdir -p /mnt/nvme/lerobot-bench/{src,ckpt,hf/hub,libero-assets}
```
From a fast host, rsync the repo `src/` (carries the optimized code), the checkpoint, and
the SmolVLM2 *small* files (config/processor/tokenizer only — the 865 MB checkpoint is
self-contained, so the 1.9 GB base weights are NOT needed):
```bash
# from the repo host:
rsync -az src/lerobot/ nvidia@<orin>:/mnt/nvme/lerobot-bench/src/lerobot/
rsync -a --partial outputs/train/sawseenvla_libero_spatial_4k_bs96_1xGPU_full_bf16/checkpoints/last/pretrained_model/ \
  nvidia@<orin>:/mnt/nvme/lerobot-bench/ckpt/
rsync -a --partial --max-size=50m \
  ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct \
  nvidia@<orin>:/mnt/nvme/lerobot-bench/hf/hub/
```
Make the checkpoint load without the base weights (it contains the full 345 `vlm.*` +
145 `lm_expert.*` tensors):
```bash
ssh nvidia@<orin> 'python3 -c "import json;p=\"/mnt/nvme/lerobot-bench/ckpt/config.json\";c=json.load(open(p));c[\"load_vlm_weights\"]=False;json.dump(c,open(p,\"w\"),indent=2)"'
```
Also copy `docker/bench_orin.py`, `docker/orin_libero_eval.sh` to `/mnt/nvme/lerobot-bench/`.

### 3.1 Base inference image — `docker/Dockerfile.orin`
Default base is `nvcr.io/nvidia/pytorch:25.06-py3-igpu`; if that's not pullable on the
link, use the dustynv Jetson build already on the device (verified: torch 2.9.1 + CUDA,
device "Orin"). Pins torch/torchvision/numpy via a generated constraints file so pip
can't replace the Jetson torch; installs `transformers==5.3.0` (SmolVLM2) + lerobot deps.
```bash
cd /mnt/nvme/lerobot-bench
docker build -f Dockerfile.orin \
  --build-arg BASE_IMAGE=pytorch:2.9.1-r36.4.tegra-aarch64-cp312-cu128-24.04 \
  -t lerobot-orin .
```

### 3.2 Max performance
```bash
echo 123 | sudo -S nvpmodel -m 0      # MAXN
echo 123 | sudo -S jetson_clocks
```

### 3.3 Latency benchmark (the 4-config opt-vs-baseline numbers)
```bash
docker run --rm --runtime nvidia --ipc=host -e HF_HUB_OFFLINE=1 \
  -v /mnt/nvme/lerobot-bench/src:/lerobot/src \
  -v /mnt/nvme/lerobot-bench/ckpt:/ckpt:ro \
  -v /mnt/nvme/lerobot-bench/hf:/hf \
  -v /mnt/nvme/lerobot-bench/bench_orin.py:/workspace/bench_orin.py \
  lerobot-orin python3 bench_orin.py /ckpt
```
Expected (MAXN, bs=1, 512², bf16): K=10 eager **734.3 ms** · K=10 two-graph 113.8 ms ·
K=1 eager **180.2 ms** · K=1 two-graph **66.8 ms** · bit-exact diff 0.000000.
(`Dockerfile.orin` sets `PYTHONPATH=/lerobot/src`, `HF_HOME=/hf`; bench_orin.py reads
`max_state_dim` from the config — Orin loaded state_dim=32, img_keys=2, L=48.)

### 3.4 LIBERO eval on Orin — `docker/Dockerfile.orin.libero` + `orin_libero_eval.sh`
**Feasibility: confirmed** — MuJoCo EGL offscreen rendering works on the Tegra GPU once
the containerized-EGL gap is closed (§3.6). Eval needs only `hf-libero` (bundles bddl +
init-states), the **`lerobot/libero-assets`** bundle (~408 MB, arch-independent), and the
checkpoint — **no training dataset**.

Stage the assets offline (from a host that already has them, e.g. `~/.cache/libero/assets`):
```bash
rsync -a --partial ~/.cache/libero/assets/ nvidia@<orin>:/mnt/nvme/lerobot-bench/libero-assets/
```
Build the libero image (self-healing loop for the lossy link — see §3.5):
```bash
cd /mnt/nvme/lerobot-bench
for i in $(seq 1 8); do
  DOCKER_BUILDKIT=1 docker build -f Dockerfile.orin.libero -t lerobot-orin-libero . && break
  echo "attempt $i dropped; resuming from pip cache mount"; sleep 5
done
```
Run a **smoke eval** first (1 task, few episodes) to validate the loop, then scale to the
full protocol. The optimized path = `SAWSEEN_CUDAGRAPH=1` + `--policy.num_steps=1`:
```bash
docker run --rm --runtime nvidia --ipc=host \
  -e HF_HUB_OFFLINE=1 -e SAWSEEN_CUDAGRAPH=1 \
  -v /mnt/nvme/lerobot-bench/src:/lerobot/src \
  -v /mnt/nvme/lerobot-bench/ckpt:/ckpt:ro \
  -v /mnt/nvme/lerobot-bench/hf:/hf \
  -v /mnt/nvme/lerobot-bench/libero-assets:/libero-assets:ro \
  -v /mnt/nvme/lerobot-bench/orin_libero_eval.sh:/workspace/orin_libero_eval.sh \
  lerobot-orin-libero bash /workspace/orin_libero_eval.sh \
    --policy.path=/ckpt --policy.device=cuda --policy.n_action_steps=10 \
    --policy.num_steps=1 --policy.compile_model=false \
    --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
    --eval.n_episodes=2 --eval.batch_size=2 --env.max_parallel_tasks=1
```
Full protocol: drop `--env.task_ids`, set `--eval.n_episodes=10 --eval.batch_size=10`
(→ 100 episodes). NOTE: a full Orin eval is long (sim + policy); confirm scope first.

> **STATUS (2026-06-25):** EGL render validated on-device, image-build recipe + asset
> staging + entrypoint all in place and committed (`d125a776`). The end-to-end SR run was
> in progress at time of writing (libero image build converging through link drops); the
> on-device SR number is the one remaining TODO.

### 3.5 Lossy-link build resilience (load-bearing on this device)
A bare `pip install` aborts the whole layer on one truncated wheel (`BrokenPipeError` /
`incomplete-download`). `Dockerfile.orin.libero` therefore uses
`RUN --mount=type=cache,target=/root/.cache/pip pip install --retries 10 --timeout 180 …`
(needs `# syntax=docker/dockerfile:1` at the top). Re-running the build **resumes** from
the cache — wrap it in a retry loop and it converges. Same idea for transfers: always
`rsync --partial`.

### 3.6 EGL-on-Tegra fix (the make-or-break gotcha)
The Tegra ships only the *vendor* `libEGL_nvidia.so.0`, not the glvnd dispatch
`libEGL.so.1` that PyOpenGL's `find_library('EGL')` needs → `AttributeError: 'NoneType'
object has no attribute 'eglQueryString'`. Closing the gap (baked into
`Dockerfile.orin.libero`):
- **apt:** `libegl1 libgles2 libglvnd0 libopengl0` (provides the glvnd `libEGL.so.1`),
  plus `libegl1-mesa-dev` + `cmake ninja-build build-essential` for the sdist-only
  `egl-probe` / `hf-egl-probe` CMake builds.
- **env:** `NVIDIA_DRIVER_CAPABILITIES=all` (so the nvidia runtime mounts the Tegra GL
  libs — `compute,utility` alone does NOT), `__EGL_VENDOR_LIBRARY_DIRS=/usr/lib/aarch64-linux-gnu/tegra-egl`
  (routes glvnd → the L4T EGL ICD `nvidia.json`), `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`,
  `MUJOCO_EGL_DEVICE_ID=0`.
- **Smoke test** (must print `EGL_RENDER_OK (128,128,3) mean <non-zero>`):
  ```python
  import os, numpy as np, mujoco
  m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="box" size=".5 .5 .5" rgba="1 0 0 1"/></worldbody></mujoco>')
  d = mujoco.MjData(m); mujoco.mj_forward(m, d)
  r = mujoco.Renderer(m, 128, 128); r.update_scene(d)
  print("EGL_RENDER_OK", np.asarray(r.render()).shape)
  ```
  (An `EGLError` at interpreter-exit teardown is benign — the render itself is what matters.)
- **`load_vlm_weights=false` + `HF_HUB_OFFLINE=1`** keep the run fully offline; the
  entrypoint writes `~/.libero/config.yaml` and symlinks the staged assets into the
  package dir BEFORE the first `import libero` (else libero prompts via `input()` and hangs).

---

## 4. File manifest
| path | purpose | commit |
|---|---|---|
| `src/lerobot/policies/sawseenvla/modeling_sawseenvla.py` | embed_suffix fix + two-graph CUDA capture | `ca2de108` |
| `src/lerobot/policies/smolvla/smolvlm_with_expert.py` | opt-in SDPA path | `ca2de108` |
| `design/SAWSEENVLA_INFERENCE_OPT.md` | analysis / ranked levers / measured results | `ca2de108`, `c716390a` |
| `docker/Dockerfile.orin` | Jetson inference image (base) | `c716390a` |
| `docker/bench_orin.py` | 4-config latency benchmark (bit-exact check) | `c716390a` |
| `docker/Dockerfile.orin.libero` | Jetson LIBERO-eval image (sim stack + EGL fix) | `d125a776` |
| `docker/orin_libero_eval.sh` | LIBERO eval entrypoint (config.yaml + offline assets) | `d125a776` |
| `design/SAWSEENVLA_OPT_REPRODUCE.md` | this runbook | — |

## 5. Gotchas checklist (what bit us)
1. Orin root FS full → stage on `/mnt/nvme`; point HF_HOME/caches there.
2. Checkpoint is self-contained → `load_vlm_weights=false`, ship only SmolVLM2 small files (saves 1.9 GB).
3. EGL on Tegra needs the **glvnd loader** + `NVIDIA_DRIVER_CAPABILITIES=all` + vendor ICD dir (§3.6).
4. Lossy link → BuildKit pip cache mount + retry loop; `rsync --partial`.
5. Two graphs, not one — a single combined CUDA graph replays stale K/V (26% SR). Verify with a 2-input diagnostic.
6. `torchcodec` is unavailable on aarch64 → eval video via `av`.
7. nvcr.io base (~15 GB) is impractical to pull on a slow link → `BASE_IMAGE` ARG falls back to the on-device dustynv Jetson torch image.
8. Device may reboot under load → NVMe staging persists; re-check `ping`/ssh.
