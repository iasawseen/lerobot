"""SawSeenVLA inference benchmark on Jetson AGX Orin.

Single process, fixed seed + fixed inputs/noise (clone per call) so eager and
graphed paths are directly comparable and bit-exactness is checkable. Measures
four configs and prints a summary + speedups:

  [A] K=10 eager        -> baseline / "not optimized" (as the model ships)
  [D] K=10 two-graph    -> CUDA-graph speedup at the SAME quality as A
  [B] K=1  eager        -> K-reduction only
  [C] K=1  two-graph    -> fully optimized (SAWSEEN_CUDAGRAPH=1)

Usage: python bench_orin.py <checkpoint_dir>
"""
import os
import statistics
import sys
import time

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from lerobot.policies.sawseenvla.modeling_sawseenvla import SawSeenVLAPolicy
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

CKPT = sys.argv[1]
DEV = "cuda"

print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}  cuda {torch.version.cuda}")
p = SawSeenVLAPolicy.from_pretrained(CKPT).to(DEV).eval()
cfg = p.config
RES = cfg.resize_imgs_with_padding[0]
L = cfg.tokenizer_max_length
ik = list(cfg.image_features)
SD = getattr(cfg, "max_state_dim", 8)
print(f"loaded: res={RES} chunk={cfg.chunk_size} vlm_layers={cfg.num_vlm_layers} "
      f"img_keys={len(ik)} state_dim={SD} L={L}")

torch.manual_seed(0)
batch = {k: torch.rand(1, 1, 3, RES, RES, device=DEV) for k in ik}
batch[OBS_STATE] = torch.rand(1, 1, SD, device=DEV)
batch[OBS_LANGUAGE_TOKENS] = torch.randint(0, 30000, (1, L), device=DEV)
batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(1, L, dtype=torch.bool, device=DEV)
noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device=DEV)


@torch.no_grad()
def call():
    p.reset()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return p.predict_action_chunk({k: v.clone() for k, v in batch.items()}, noise=noise.clone())


@torch.no_grad()
def bench(nw=10, ni=40):
    lat = []
    out = None
    for i in range(nw + ni):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = call()
        torch.cuda.synchronize()
        if i >= nw:
            lat.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(lat), (statistics.stdev(lat) if len(lat) > 1 else 0.0), out


def reset_graph():
    for a in ("_infer_cg", "_denoise_cg"):
        if hasattr(p.model, a):
            delattr(p.model, a)


res = {}

# [A] K=10 eager — baseline / not optimized
cfg.num_steps = 10
os.environ["SAWSEEN_CUDAGRAPH"] = "0"
reset_graph()
mA, sA, oA = bench()
res["K10_eager"] = (mA, sA)
print(f"[A] K=10 eager   (baseline)  : {mA:7.1f} +/- {sA:4.1f} ms   ({1000.0/mA:.2f} chunks/s)")

# [D] K=10 two-graph — graph speedup at same quality
os.environ["SAWSEEN_CUDAGRAPH"] = "1"
reset_graph()
mD, sD, oD = bench()
res["K10_graph"] = (mD, sD)
dD = (oA - oD).abs().max().item()
print(f"[D] K=10 two-graph           : {mD:7.1f} +/- {sD:4.1f} ms   ({1000.0/mD:.2f} chunks/s)  "
      f"diff_vs_A={dD:.6f}  speedup={mA/mD:.2f}x")

# [B] K=1 eager — K-reduction only
cfg.num_steps = 1
os.environ["SAWSEEN_CUDAGRAPH"] = "0"
reset_graph()
mB, sB, oB = bench()
res["K1_eager"] = (mB, sB)
print(f"[B] K=1  eager               : {mB:7.1f} +/- {sB:4.1f} ms   ({1000.0/mB:.2f} chunks/s)")

# [C] K=1 two-graph — fully optimized
os.environ["SAWSEEN_CUDAGRAPH"] = "1"
reset_graph()
mC, sC, oC = bench()
res["K1_graph"] = (mC, sC)
dC = (oB - oC).abs().max().item()
print(f"[C] K=1  two-graph (OPTIMIZED): {mC:7.1f} +/- {sC:4.1f} ms   ({1000.0/mC:.2f} chunks/s)  "
      f"diff_vs_B={dC:.6f}  speedup={mB/mC:.2f}x")

print("=== SUMMARY (AGX Orin) ===")
print(f"baseline  K=10 eager      : {mA:7.1f} ms")
print(f"optimized K=1  two-graph  : {mC:7.1f} ms")
print(f"END-TO-END SPEEDUP (opt vs baseline): {mA/mC:.2f}x   ({mA:.1f} -> {mC:.1f} ms)")
print(f"  of which K-reduction (K10->K1 eager): {mA/mB:.2f}x")
print(f"  of which CUDA-graph    (K1 eager->graph): {mB/mC:.2f}x")
print(f"bit-exactness: K10 A-vs-D={dD:.6f}  K1 B-vs-C={dC:.6f}  (mean|action|={oC.abs().mean().item():.4f})")
print("done")
