# SawSeenVLA — Inference & Deployment Optimization

*Companion to `design/TRAIN_SPEED_UP.md` (training-throughput). This doc is inference/deployment-only. Where a lever overlaps with training, it is cross-referenced, not repeated.*

Target object: ~450M VLA — frozen SmolVLM2-500M backbone (first 16 of 32 layers, PixelShuffle → 64 vision-tokens/frame, `self_attn_every_n_layers=2`) + flow-matching action expert, K=10 Euler denoise, chunk_size=50, bf16. Serving targets: RTX 3090/3090 Ti (sm_86, dev + desktop) and Jetson AGX Orin 64 GB (sm_87, on-robot). Both Ampere ⇒ **bf16/fp16/INT8 tensor cores, no FP8, no useful hardware INT4.** Every recommendation is constrained to that precision envelope.

---

## 1. Executive summary — ranked levers against the 218 ms / 450M baseline

The measured profile is unambiguous: **the K=10 denoise loop is 179 ms of the 218 ms full-chunk latency (82%), near-linear in K; the VLM prefix is a one-time ~39 ms (18%).** This single fact reorders every conventional VLM-serving instinct — quantizing/shrinking the VLM attacks 18%; cutting denoise steps attacks 82%.

| # | Lever | Target | Latency payoff (bs=1) | Memory payoff | Accuracy risk | Cost | Platform |
|---|---|---|---|---|---|---|---|
| 1 | **Flow-head distillation to 1–2 NFE** (SnapFlow-style self-distill) | 82% | **218 → ~57 ms (K=1), −74%**; K=2 measured 74 ms (−66%) | none | ~neutral (+1 pt LIBERO reported on the teacher class); a few pts on hardest tasks at K=1 | ~12 GPU-h, ~few-hundred LOC | both |
| 2 | **Static KV-cache + CUDA-graph the denoise step** | 82% | removes Python/launch overhead × K; large at bs=1 — pending profile | none | none | medium (StaticCache prerequisite) | both |
| 3 | **SDPA/FlashAttention-2 in the expert** (replace eager fp32-upcast attn) | 82%+18% | cuts the dominant per-step matmul, both prefix & denoise | small | none | low | both |
| 4 | **Higher-order / scheduled ODE solver** (Heun, DPM-Solver, non-uniform t) | 82% | K=10 → ~4–6 at matched quality, **~1.5–2.5×**, training-free | none | ~0 (ceiling ≈ K≈4) | very low | both — do first |
| 5 | **INT8-via-TensorRT (expert MLP/DiT blocks)** | 82% | conditional; **benchmark per-block — INT8 can regress vs FP16 on transformer/control-flow graphs** | weights ½ | validate closed-loop | high | Orin first |
| 6 | **INT8/INT4 weight-only on the VLM** | 18% | **~0 latency (often negative)** | **2–4× VLM weights** — the Orin unified-pool lever | ~0 (INT8) / small (INT4-AWQ) | low (bnb/AWQ) | Orin |
| 7 | **RTC tuning** (already built, untuned) | hides latency | overlaps compute with execution; ~20% faster motion at deploy | none | neutral | low (config sweep) | both |
| 8 | **Multi-env / sampling batching** | throughput | bs=10 = 30 ms/sample, **7.2×** | linear | none | none | 3090, multi-robot |

Stacking note: #1, #2, #3, #4 operate on different graph regions and **multiply**. #6 is orthogonal (memory). #1 and an OFT-style L1 head are *mutually exclusive* routes to one forward pass — pick one (keep flow + distill for multimodal contact-rich tasks).

**The one-line strategy:** distillation/step-reduction is the *latency* lever (82%); quantization is the *memory* lever (Orin unified pool). Conflating them is the standard mistake this profile rules out.

---

## 2. The latency profile — where to spend effort

Measured (reference SawSeenVLA, 3090 Ti, bf16, bs=1, 512² imgs):

- Full chunk, K=10: **218 ms (4.6 chunks/s).**
- Decomposition: VLM prefix (image embed + forward, **once per chunk**) ~39 ms = **18%**; per denoise step ~17.9 ms; denoise loop K=10 = **179 ms = 82%.**
- K-sweep (near-linear): K=1 57 / K=2 74 / K=5 128 / K=10 218 ms.
- Batch: bs=10 = 302 ms total = **30 ms/sample (7.2× throughput)** — prefix and expert both parallelize across batch.
- Amortized control: chunk=50 → 4.4 ms/step = 229 Hz; eval `n_action_steps=10` → 21.8 ms/step = 46 Hz. **Not inference-bound in sim.**

Three structural facts drive the rest of the doc:

1. **The prefix is already amortized** (`sample_actions:836`, `fill_kv_cache=True`) — it runs once and the K denoise steps re-read its cached K/V (`denoise_step:904`, `fill_kv_cache=False`). So the 18% is already a one-time cost; there is no per-step prefix recompute to eliminate.
2. **The denoise step is a fixed-shape, repeated small-kernel launch** — K identical forwards of the action suffix through the expert. At bs=1 this is the canonical launch-overhead-bound regime (CUDA graphs / `reduce-overhead`).
3. **The integrator is fixed first-order Euler** (`sample_actions:845`, `dt=-1/num_steps`; `:876`, `x_t = x_t + dt*v_t`). Both the step *count* and the step *method* are open levers.

Because sim eval is already 46–229 Hz amortized, **single-stream latency only becomes binding on-robot (Orin) and for reactive/contact-rich tasks** where the 200–500 ms plan horizon matters. The 3090 work is mostly about throughput (multi-env eval, sampling-based selection) and serving headroom.

---

## 3. Tier 1 — denoise-loop reduction (the dominant lever, 82%)

Framing constraint that orders this entire tier: **a trained K=10 flow expert already exists.** That sharply favors *distillation-of-an-existing-teacher* over *train-from-scratch* — the latter discards sunk pretraining and re-incurs it. So from-scratch one-step trainers (MeanFlow, native shortcut, MeanFlow-VLA) are deprioritized despite strong standalone one-step numbers, and methods that fine-tune the existing expert are promoted.

### 3.1 Higher-order / scheduled ODE solver — do this first, it's free

**Mechanism.** Replace first-order Euler (`x_t += dt*v_t`) with a 2nd-order Heun (2 NFE/step, curvature-corrected) or an exponential-integrator multistep (DPM-Solver/DEIS/UniPC), and/or a non-uniform timestep schedule concentrating steps where the velocity field curves. SawSeenVLA's Beta(1.5,1.0) timestep sampling front-loads toward t→0; a matched non-uniform inference schedule is the cheapest variant.

**Payoff.** Training-free, reuses the exact expert. Realistic K=10 → ~4–6 NFE at matched quality ⇒ **1.5–2.5×** on the denoise loop. **Hard ceiling ≈ K≈4**: flow-matching velocity fields are comparatively straight in time, so high-order gains diminish — "even a simple Euler integrator can already achieve excellent performance, and the benefits of applying high-order samplers such as DPMSolver diminish considerably" [A-FloPS]. (Counterpoint: model predictions average over distributions and produce curved trajectories in practice, so the win is real but modest — [Diffusion Meets Flow Matching, qualitative, unverified as a hard number].)

**Apply to SawSeenVLA.** Generalize the loop in `sample_actions` to a pluggable integrator; keep the Beta schedule for any distillation branch but sweep inference-time t-spacing. **~50–150 LOC, 0 GPU-h.** Build this as the eval harness that scores every Tier-1 method below.

**Verdict: rank #4 overall, #1 to execute** — best ROI per engineer-hour, low ceiling. Try-before-you-distill.

### 3.2 SnapFlow-style progressive self-distillation — the headline

**Mechanism.** Train the single existing network on a mixture of (a) standard flow-matching samples (= SawSeenVLA's current MSE velocity loss, `u_t = noise − action`) and (b) **consistency samples** whose targets are two-step Euler shortcut velocities from the model's own *marginal* predictions. A **zero-initialized target-time embedding** lets one network distinguish the two objectives. No external teacher, no architecture change.

**Measured (on the directly analogous object — π0.5 3B flow VLA, LIBERO 4 suites):** 1-step **98.75% vs 97.75% K=10 teacher** (net gain); end-to-end **274 → 83 ms (3.3×)**, **9.6× denoise speedup**. On SmolVLA-500M: action MSE −8.3%, cosine +6.9%, end-to-end **3.56×**. The paper independently reports "denoising accounts for ~80% of inference" — matching SawSeenVLA's 82%, which is what makes the transfer argument tight. Adoption: **~12 h on one A800, 30k steps, VLM frozen, only the expert + target-time MLP train.**

**Apply to SawSeenVLA.** The FM branch *is* the existing loss — adoption = add the consistency branch + zero-init target-time embedding and keep training. One mismatch to sweep: SawSeenVLA's Beta(1.5,1.0) is mildly *favorable* for one-step (one-step quality is dominated by behavior near the data end t→0), but verify the consistency-branch time distribution empirically.

**Mapped payoff:** K=10 → K=1 ⇒ **218 → ~57 ms (−74%)** measured at the K-sweep; with the prefix unchanged at 39 ms, the residual is the prefix (then attack it with Tier-2/3).

**Verdict: rank #1.** The literal template for "distill an existing K=10 flow VLA to 1-step," with measured neutral-to-positive success.

### 3.3 Fallbacks and chaining

- **OneDP (distribution-matching distillation).** Frozen teacher + trainable score net, reverse-KL/score-difference loss. **1.5 → 62 Hz (42×)** on a diffusion policy; matches/beats teacher in sim (0.843-S vs 0.829 teacher) and real; **+2–10% pretraining cost.** Heavier than SnapFlow (second trainable net, GAN-like dynamics) and **[unverified at VLA scale]** — demonstrated on tens-of-M-param diffusion policies. Rank #2, fallback if self-distillation drifts. (IMLE-based distribution distillation, *From Flow to One Step* [arXiv 2603.09415], is a GAN-instability-free alternative worth noting.)
- **Consistency Policy (CTM).** 1-step or **3-step "chaining"** (refine at t_{2N/3}, t_{N/3}). The cleanest 1-vs-3-step trade curve in the literature: 3-step ≥ teacher on Square/Can; 1-step costs a few pts on the hardest (Tool Hang .70 vs teacher .79). Two tricks that matter: variance-reduced init N(0,1/T²) and dropout specifically at the s→0 step (its removal dropped Square .92→.86). **Keep a 2–3 step chaining mode** for the hardest tasks to recover what the 1-step model loses. Rank #3.

### 3.4 Avoid (from-scratch — discards your expert)

MeanFlow / MeanFlow-VLA (8.7× vs SmolVLA, but **measured against SmolVLA's baseline, not SawSeenVLA's own K=10 Euler loop**), MP1, native shortcut-from-scratch. Use only if you were retraining the head anyway. The shortcut-model self-consistency *recipe* reaches SawSeenVLA via SnapFlow rather than directly.

### 3.5 How far to cut, and the success cost

| Target | Path | Expected success cost |
|---|---|---|
| K=10 → 4–6 | Heun/DPM-Solver or schedule, **no retrain** | ~0 |
| K=10 → 3 | CTM 3-step chaining | near-zero; can match/exceed teacher |
| K=10 → 2 | SnapFlow/OneDP, keep a refinement step | small (≈0–2 pts) |
| K=10 → 1 | SnapFlow (primary) / OneDP | near-zero on well-distilled tasks (+1 pt LIBERO reported); a few pts on hardest |

The 10→2 cut is essentially free; the last step 2→1 is where multimodal tasks lose a few points — recovered by 3-step chaining or one OneDP refinement.

---

## 4. Tier 2 — runtime & kernels

### 4.1 Static KV-cache + CUDA-graph the denoise step — rank #2 overall

**Why this is the structural match.** At bs=1 the GPU finishes each small expert kernel faster than the CPU can dispatch the next, so the 17.9 ms/step is partly `cudaLaunchKernel` + Python overhead, not FLOPs. The denoise loop is **static-shape, static-K, no growing KV** (the prefix is fixed at `fill_kv_cache=True` and never extends during the loop) — strictly *easier* to graph-capture than an LLM decode. The most directly analogous precedent is Vrushank Desai's diffusion-policy denoise loop **on an RTX 3090**: **~3.4× over eager, ~2.65× over torch.compile**, "the vast majority" from CUDA graphs (+ a custom Conv1d kernel — the conv half won't transfer to a transformer expert, so treat the transferable win as the launch-overhead removal alone). PyTorch's own case studies show graphed sub-regions ~5× faster, gains largest at small batch.

**Prerequisite (blocking).** The cache is currently dynamic — `smolvlm_with_expert.py:276` does `torch.cat([past_key_values[...], key_states])` on the `fill_kv_cache=False` path. The in-code TODO (`:272`) names the fix: preallocate a `StaticCache` with `max_len` declared up front (one cudaMalloc), no per-step `torch.cat`. Static shapes are a hard requirement for graph capture. **This is the unlock for both manual `torch.cuda.graph` capture of the whole K-loop and `torch.compile(mode="reduce-overhead")`.**

**Apply.** (1) Land StaticCache. (2) `torch.compile(model, mode="reduce-overhead", fullgraph=True)` over the expert forward → auto CUDA graphs + RoPE/RMSNorm fusion; escalate to `max-autotune` for the deployed build. (3) If overhead persists, manually capture the K-step loop (static input/output buffers, side-stream warmup; cache VLM prefix K/V before capture; pass `getCurrentCUDAStream()` to any custom kernel or capture silently breaks).

**Gate.** Profile first (Nsight Systems): confirm how much of 17.9 ms is CPU-side gaps vs compute. **If the step is compute-bound, the ranking inverts toward precision/distillation** and CUDA graphs help far less. This is the explicit go/no-go.

Cross-ref: `TRAIN_SPEED_UP.md` covers `reduce-overhead`/CUDA-graphs for *training* and the `pad_language_to=max_length` requirement for compile shape stability. Note its finding that **eval should pass `compile_model=false`** — warmup doesn't amortize over short rollouts; for a *deployed long-lived server* the AOT/static-graph path (below) is the right answer instead.

### 4.2 Attention: replace eager fp32-upcast with SDPA/FA-2 — rank #3

`get_attention_interface` returns `eager_attention_forward` (fp32-upcast matmul) on **both** prefix and **every denoise step**. SDPA (dispatching to a fused/FA-style kernel) or FlashAttention-2 (sm_86/sm_87, fp16/bf16) cuts the dominant per-step attention. The action-suffix sequence is short, so the *per-step* win is modest, but it lands on 82% of the budget × K and on the prefix. **FlashAttention-3 is Hopper-only (WGMMA/TMA/FP8) — does not transfer to Ampere; drop it.** Low LOC, no retrain.

### 4.3 AOTInductor / torch.export — deployment hygiene

`torch.export` → `aoti_compile_and_package()` → `.pt2` (shared `.so` + cubins), loadable from C++ with **no Python at runtime**. Same Inductor kernels as 4.1 (no extra throughput), but strips JIT warmup from the robot binary — the clean answer to the "eval shouldn't pay compile warmup" finding. On Orin you'll often prefer the TensorRT engine (§7) for raw speed; AOTInductor is the path when staying in the PyTorch stack. Pin the PyTorch version (historically prototype-labeled).

### 4.4 channels-last

Free-ish only for conv/vision parts; the pure-transformer expert sees ~nothing. Apply `to(memory_format=torch.channels_last)` to the SigLIP tower; skip for the expert.

---

## 5. Tier 3 — quantization & memory (a memory lever, not latency)

**State this once, plainly:** weight-only quantization on the VLM is a **memory** win, **not** a latency win — the VLM is 18% of latency and you are not VLM-latency-bound. INT8/INT4 buy footprint (the Orin unified-pool lever) at near-zero accuracy cost; they do **not** speed up single-stream serving.

### 5.1 The OpenVLA warning label

OpenVLA-7B Bridge (Table 2): **bf16 71.3% / 16.8 GB → INT8 58.1% / 10.2 GB → INT4 71.9% / 7.0 GB.** INT8 was both *less accurate and slower* than INT4 and bf16 — weight-only LLM.int8()-style quant added dequant overhead without enough bandwidth savings to pay it back; INT4's larger bandwidth reduction more than offset its overhead. Caveat: that pathology is partly **autoregressive-decode-specific and A5000-specific** — SawSeenVLA's flow head has no token-by-token decode, so it may not reproduce; verify on-stack. The transferable lesson: **don't assume INT8 > INT4, and don't expect latency from weight-only quant.**

### 5.2 Scale-specific guidance for a sub-1B VLM

- The LLM.int8() emergent-outlier phase transition is at **~6.7B params** (verified) — a sub-1B VLM **does not need the mixed-precision outlier machinery**; plain per-channel INT8 should be clean. But small models have less redundancy, so **validate W8A8 on-task**, not on big-model PTQ numbers.
- 4-bit weight-only: prefer **AWQ** (activation-aware, no calibration overfitting → modality-robust for robot observations, which differ from text) or **NF4** (bnb, on-the-fly, lowest adoption cost). GPTQ/AWQ stay <4–6% perplexity hit on ≥7B; **expect somewhat more on a 450M VLM** — validate.
- For *recovery* (not gains), QAT à la **SQIL/Saliency-Aware Quantized Imitation Learning** (ICCV 2025): OpenVLA INT4 **73.2% vs 73.8% bf16** (recovers within ~1 pt, does **not** exceed), **up to 2.5× speedup / 2.5× energy** on Orin. (The earlier-circulated "84.4 vs 83.6, beats baseline" figures are not in the source — do not cite.)
- **BitVLA** ternary {−1,0,1}+INT8: **94.8% LIBERO vs 97.1% (−2.3 pts), 1.4 vs 15.4 GB (11×), ~4.6× end-to-end latency reduction** (relative; the absolute 73 ms/341 Hz figures are not in the paper — drop them). The memory ceiling if you go all-in, at a training-time cost.

### 5.3 INT8 of the *expert* — the only quant with latency ROI here

Since the expert is 82%, the question that actually matters is whether the per-step expert matmuls run W8A8 on Ampere tensor cores. **W8A8 via TensorRT (SmoothQuant-style activation handling)** is the only Ampere path that speeds up matmuls (vs weight-only which just dequants). Two real risks: (1) **INT8 frequently does *not* beat FP16 on transformer/control-flow graphs** — documented Orin regressions (e.g. a ViT-S+DPT 2.7× regression; MiDaS DPT FP16 173 vs INT8 97 FPS) where small/odd-shape layers fall off the INT8 fast path; (2) **quant error compounds across the K-step integrator.** Mitigation: INT8 the expert MLP/DiT *blocks*, keep the ODE solver loop static and higher-precision, **A/B per-block, keep INT8 only where it beats FP16.** Start FP16, treat INT8 as an empirical experiment.

### 5.4 KV-cache quant — a non-lever

A VLA emits a short action chunk; the KV cache is tiny (cached prefix + short suffix). KV-cache quant targets long autoregressive decode you don't have. One sentence in the doc, no workstream.

---

## 6. Serving & async

### 6.1 Re-plan frequency — the free correctness win (config only)

SmolVLA Table 13 (LIBERO): replan every **1 step 80.3% / 10 steps 82.8% / 50 (open-loop full chunk) 51.8%.** **Open-loop chunk execution is the dominant failure mode, not inference latency.** Set the equivalent of `chunk_size_threshold` g≈0.5–0.6 (LeRobot default 0.7; docs recommend 0.5–0.6), verify the action queue never empties. Zero code.

### 6.2 RTC — built but untuned

`rtc_config` + `RTCProcessor` are wired into `sample_actions` (`:860`, swapping `rtc_processor.denoise_step` into the loop): freeze the first *d* actions guaranteed to execute, soft-inpaint (LINEAR `prefix_attention_schedule`, `max_guidance_weight=10.0`, `execution_horizon=10`) the rest, so the next chunk computes while the current executes. Measured elsewhere: tolerates >100/200/300 ms injected delay, ~20% faster motion, beats temporal ensembling and BID (BID costs 2.3× RTC latency). **The gap: `execution_horizon` / `inference_delay` / guidance schedule have no deployment sweep on this rig.** Run that sweep — it's a config workstream, not code.

**Open interaction to stress-test (do not assume clean stacking):** RTC's inpainting relies on iterative denoising budget (it uses ~5 steps). A 1-step distilled model (§3.2) has **no iterative budget** for the inpainting refinement RTC needs. Whether RTC degrades at K=1–2 is unresolved — *Action-Prior Denoising for Smooth Real-Time Chunking* and FASTER are evidence this is an active subproblem. **Validate RTC + distilled-K jointly; if it breaks, keep K=2–3 specifically to preserve RTC's inpainting budget.**

### 6.3 Cross-chunk prefix KV reuse & pipelining

Within-chunk reuse is done (`fill_kv_cache`). The incremental opportunity is **cross-chunk**: with a fixed instruction the text/state tokens' K/V are exactly reusable; **image-token K/V are not** (frames change). VLA-Cache-style per-token change detection reuses the static-region KV (**1.63× LIBERO, 27% FLOPs, −0.3% SR, training-free** on OpenVLA/CogACT) — manipulation-native because most of the visual field is static between control steps. Pipelining (double-buffer: prefill obs t+1's prefix while the K-loop for chunk t runs) pairs with high replan frequency. Medium effort.

### 6.4 Multi-robot / sampling batching — the measured 7.2×

bs=10 = 30 ms/sample. Relevant for multi-env eval, sampling-based action selection, or multi-robot serving — **not single-robot latency.** For many-robot-per-GPU serving, OxyGen-style deadline-aware continuous batching reports large throughput gains, but **those magnitudes were measured on Jetson Thor (Blackwell, FP8) — the mechanism transfers to Ampere, the magnitudes do not.** Skip Spec-VLA (autoregressive-discrete only; no flow analog) and temporal ensembling (superseded by RTC).

---

## 7. Edge deployment — Jetson AGX Orin 64 GB

**The compiled path (both targets):** PyTorch → ONNX / `torch.export` → TensorRT engine. INT8 on Orin (GPU tensor cores **and** the 2× NVDLA v2, both INT8/FP16-only), bf16/INT8 on the 3090. **Build/calibrate the engine on the target device** — INT8 calibration results vary across devices. **FP8 is confirmed unavailable on Orin (sm_87 < sm_89); drop it.** A proven competing on-Orin path is **GGUF/llama.cpp INT4** (LiteVLA-Edge hits 150.5 ms via Q4_K_M GGUF, *not* TensorRT) — worth A/B'ing against TRT for the VLM.

**Memory budget.** Unified LPDDR5 (204.8 GB/s) shared by model + activations + OS + camera + ROS 2 stack ⇒ budget against **~55 GB usable, not 64.** A 450M model is ~0.9 GB bf16 / ~0.45 GB INT8 — **memory is not the binding constraint on Orin 64 GB; latency is.** (Footprint only bites on the 8 GB Orin Nano or if you stack a world-model buffer.) The 24 GB 3090 has ample headroom for 450M; quantization there is optional.

**Realistic latency/power for the 450M flow loop on Orin.** Anchors: π0 (~3B) measures **920.6 ms / 1.09 Hz / 1.867 kJ per inference** on AGX Orin; SmolVLA-450M-class runs **~2 Hz** raw; vla.cpp times SmolVLA at **28.16 ms/step on RTX 3060 and 141.81 ms/step on Orin Nano** (≈5× the desktop per-step). Read-across to the reference 3090-Ti 17.9 ms/step: expect roughly **~70–100 ms/step on AGX Orin** at the full power budget ⇒ **K=10 ≈ 0.7–1.0 s/chunk raw.** That is sub-real-time per-replan; it is made deployable by (a) **action chunking** (one chunk ≈ 0.2–1.0 s of motion executed at 30–50 Hz controller rate) and (b) **async** (overlap next inference with current execution). **Tier-1 distillation to K=1 collapses that to ~120–250 ms/chunk on Orin** — the difference between "sub-real-time" and "responsive."

**Power.** 15 / 30 / 50 W / MAXN ~60 W. MAXN ~doubles the rate at ~4× the watts; **30–50 W is the realistic sustained envelope on a battery platform.** Energy/inference (1.867 kJ for π0) is the honest edge metric — distillation cuts it ~linearly with K.

**DLA viability.** The 2× NVDLA v2 are INT8/FP16-only and **cannot run transformer/attention layers** (JetPack 6.2) — the expert runs on the Orin GPU via TRT; DLA is only useful to offload conv/vision sub-nets. Don't plan the expert on DLA.

**Net Orin recipe:** sub-1B flow VLA, **distill to K=1–2 (Tier-1)**, action chunking + async + tuned RTC, bf16 baseline first, then per-block INT8-via-TRT on the expert **only where it beats FP16** (validate closed-loop success, not per-token error), INT4-AWQ on the VLM for the unified-pool memory win.

---

## 8. Architecture-level — and whether it stacks with Tier 1

All training-free unless noted; all attack a *different* graph region than the denoise loop, so they **multiply** with Tier 1 — but applied to an already-compressed SawSeenVLA (64 tokens, 16 layers) expect **smaller relative gains than the papers** (which start from 256–576 tokens / full depth).

- **Vision-token pruning below 64.** Action-aware variants only (ADP 1.35×; VLA-Pruner up to 1.99× at 12.5% retention — both training-free, condition retention on what the action head attends to). Generic ToMe doesn't know which tokens matter for manipulation. Realistically a further **1.2–1.4× on the prefill** from 64, not 2×. Attacks the 18%. Risk: over-pruning hurts fine/contact-rich phases.
- **Deeper layer cut below 16.** "Finetuning VLAs Requires Fewer Layers Than You Think" (CLP): SmolVLA **16→6**, π0 18→12, flat to ~50% prune — but **needs a light finetune** to restore latent geometry (not zero-shot below ~12). Also halves the DiT action-head depth (16→8), which shrinks per-step expert cost — **stacks directly on the 82%.**
- **Resolution 512→384→256.** PixelShuffle fixes output at 64 tokens/frame *regardless of input res*, so dropping resolution does **not** change the LLM token count — it only cuts SigLIP encoder FLOPs (encoder-only win, modest vs the denoiser). Risk: small-object/contact precision. Cross-ref `TRAIN_SPEED_UP.md` which covers res down-scaling for *training* throughput.
- **Parallel-decode / L1 regression head (OFT).** An *alternative* to iterative denoising, not a complement. SawSeenVLA's flow head already captures OFT's continuous-action benefit; the real fork is **distill flow to 1-NFE (§3.2) vs swap to an L1 head** — same single-pass endpoint, but **flow+distillation preserves multimodal/contact-rich action distributions better.** Keep flow.
- **MoE.** Capacity-at-fixed-active-compute, not a cheap latency cut on an already-truncated 450M model; routing instability + retraining cost. Deprioritize. KV-eviction (H2O/StreamingLLM) is decode-phase, mismatched to a prefill-heavy VLA — prefer VLA-Cache/SnapKV.

---

## 9. Recommended roadmap (ROI order)

| # | Experiment | Hypothesis | Cost | Latency Δ vs 218 ms | Mem Δ vs 450M | Risk to 75% LIBERO-spatial | Platform |
|---|---|---|---|---|---|---|---|
| 1 | **Pluggable solver + non-uniform schedule** (Heun/DPM) | K=10→~4–6 free | ~100 LOC, 0 GPU-h | 218 → ~100–130 ms | 0 | ~0 (ceiling K≈4) | both |
| 2 | **StaticCache + CUDA-graph denoise step** (gate on Nsight) | kill launch overhead × K at bs=1 | ~1 wk, medium LOC | per-step ↓ (profile-dependent) | 0 | 0 | both |
| 3 | **SDPA/FA-2 in expert** (drop eager fp32 attn) | cut dominant per-step matmul | low LOC | per-step ↓ | small | 0 | both |
| 4 | **SnapFlow self-distill → K=1–2** | 82% → near-1-step, SR-neutral | ~12 GPU-h, few-hundred LOC | **218 → ~57 ms (K=1) / 74 ms (K=2)** | 0 | small at K=1 (keep K=2–3 chaining fallback) | both |
| 5 | **RTC deployment sweep** (+ joint w/ distilled K) | hide latency; verify inpaint budget at low K | config, ~days | hides, ~20% faster motion | 0 | neutral; **flag K=1+RTC interaction** | both |
| 6 | **Re-plan freq g≈0.5–0.6** | avoid 51.8% open-loop cliff | config | none (correctness) | 0 | **+** (avoids cliff) | both |
| 7 | **INT4-AWQ on VLM** | Orin unified-pool memory | low (bnb/AWQ) | ~0 (maybe negative) | **VLM weights 4×** | ~0 (validate) | Orin |
| 8 | **Per-block INT8-via-TRT on expert** | the only quant with latency ROI | high, on-device calib | conditional; **A/B vs FP16** | weights ½ | validate closed-loop | Orin |
| 9 | **CLP layer cut 16→~8 + DiT-head halving** | shrink prefix+per-step | light finetune | both regions ↓ | weights ↓ | needs finetune to hold SR | both |
| 10 | **Action-aware token prune <64 (ADP/VLA-Pruner)** | ~1.2–1.4× prefill | training-free | prefix ↓ | small | over-prune hurts fine phases | both |

**Per-platform target-latency table.**

| Stage | 3090 Ti (bs=1) | AGX Orin (est., full power) |
|---|---|---|
| Baseline K=10 | 218 ms (4.6 chunks/s) | ~0.7–1.0 s/chunk (~1–1.5 Hz) |
| + Tier-1 solver (K≈5) | ~128 ms | ~0.4–0.5 s/chunk |
| + SnapFlow K=1 | **~57 ms (17.5 chunks/s)** | **~120–250 ms/chunk (~4–8 Hz)** |
| + CUDA-graph/SDPA/INT8-expert | **<40 ms** (overhead removed) | **<150 ms/chunk**, lower power | 

With chunking + async, both rows are well above controller rate for manipulation; the Orin K=1 row is what makes contact-rich/reactive tasks (200–500 ms plan horizon) tractable on-robot.

**Sequencing logic:** #1/#3/#6 are same-day, zero-risk. #2 gated on Nsight. #4 is the headline and should start in parallel with #1 (use #1 as its eval harness). #5 must be validated *jointly* with #4 (the RTC-at-low-K open question). #7/#8 are Orin-only memory/edge plays — #8 strictly A/B'd per-block.

---

## 10. References

**Few-step / distillation**
- SnapFlow — One-Step Action Generation for Flow-Matching VLAs via Progressive Self-Distillation — https://arxiv.org/abs/2604.05656
- OneDP — One-Step Diffusion Policy (distribution-matching distillation) — https://arxiv.org/abs/2410.21257
- Consistency Policy (CTM) — Prasad et al. — https://arxiv.org/abs/2405.07503
- Shortcut Models — Frans, Hafner, Levine, Abbeel — https://arxiv.org/abs/2410.12557
- MeanFlow — https://arxiv.org/abs/2505.13447 · Mean-Flow One-Step VLA — https://arxiv.org/abs/2603.01469 · MP1 — https://arxiv.org/abs/2507.10543 *(from-scratch — deprioritized)*
- One-Step Flow Policy (OFP) — https://arxiv.org/abs/2603.12480
- From Flow to One Step (IMLE distribution distillation) — https://arxiv.org/abs/2603.09415
- FASTER — Rethinking Real-Time Flow VLAs — https://arxiv.org/abs/2603.19199
- Rectified Flow / Flow Straight and Fast — https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html · Rectified Diffusion: Straightness Is Not Your Need — https://arxiv.org/abs/2410.07303
- DPM-Solver — https://arxiv.org/abs/2206.00927 · A-FloPS — https://arxiv.org/abs/2509.00036 · Diffusion Meets Flow Matching — https://diffusionflow.github.io/ *[qualitative claim unverified]*

**Runtime / kernels**
- Vrushank Desai — Diffusion-Policy Inference Optimization (CUDA graphs on RTX 3090, ~3.4×/2.65×) — https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-ix---putting-it-all-together · Part VIII (custom kernel + CUDA graphs) — https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-viii---integrating-a-custom-cuda-kernel-cuda-graphs-in-pytorch
- PyTorch — Accelerating PyTorch with CUDA Graphs — https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/ · GPT, Fast (compile + static cache, 4.2×) — https://pytorch.org/blog/accelerating-generative-ai-2/
- NVIDIA — CUDA-Graph diffusion best practice (SD v2 full-iteration capture) — https://docs.nvidia.com/dl-cuda-graph/examples/stable-diffusion-v2.html
- FlashAttention (FA-2 Ampere; FA-3 Hopper-only) — https://github.com/Dao-AILab/flash-attention · FA-3 — https://arxiv.org/abs/2407.08608
- AOTInductor — https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html · Torch-TensorRT — https://docs.pytorch.org/TensorRT/

**Quantization / edge**
- OpenVLA (quant Table 2; INT8-slower-than-INT4) — https://arxiv.org/abs/2406.09246
- SmoothQuant — https://arxiv.org/abs/2211.10438 · LLM.int8() — https://arxiv.org/abs/2208.07339 · GPTQ — https://arxiv.org/abs/2210.17323 · AWQ — https://arxiv.org/abs/2306.00978 · QLoRA/NF4 — https://arxiv.org/abs/2305.14314
- SQIL — Saliency-Aware Quantized Imitation Learning (Orin INT4 2.5×, recovers within ~1 pt) — https://arxiv.org/abs/2505.15304 · QAIL — https://arxiv.org/abs/2412.01034
- BitVLA — https://arxiv.org/abs/2506.07530 *(−2.3 pts, 11× memory; absolute ms/Hz figures not in paper — dropped)*
- FP8 vs INT8 for inference (Qualcomm) — https://arxiv.org/abs/2303.17951 · TensorRT INT8-slower-than-FP16 on control-flow — https://forums.developer.nvidia.com/t/tensorrt-int8-inference-is-slower-than-fp16-in-models-with-conditional-flow/294653 · Jetson Edge-LLM (FP8 N/A on Orin) — https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/
- torchao — https://github.com/pytorch/ao · vLLM LLM Compressor (W8A8) — https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w8a8_int8/
- Characterizing VLA Models across XPUs (π0 Orin 920.6 ms / 1.867 kJ) — https://arxiv.org/abs/2604.24447 · LiteVLA-Edge (150.5 ms Q4_K_M GGUF on Orin) — https://arxiv.org/abs/2603.03380 · NanoVLA — https://arxiv.org/abs/2510.25122

**Serving / async / architecture**
- SmolVLA (replan ablation 80.3/82.8/51.8; async 30%/2×) — https://arxiv.org/abs/2506.01844 · LeRobot async docs — https://huggingface.co/docs/lerobot/async
- Real-Time Chunking — https://arxiv.org/abs/2506.07339 · BID (2.3× RTC latency) — https://arxiv.org/abs/2408.17355 · Training-time RTC — https://arxiv.org/abs/2512.05964 *[hard numbers unverified]*
- OxyGen (Thor-class — magnitudes not Ampere-transferable) — https://arxiv.org/abs/2603.14371 · vla.cpp (SmolVLA 28.16 ms RTX 3060 / 141.81 ms Orin Nano) — https://arxiv.org/abs/2606.08094 · Spec-VLA (AR-only) — https://arxiv.org/abs/2507.22424
- OpenVLA-OFT (parallel decode, 26×, 97.1% LIBERO) — https://arxiv.org/abs/2502.19645 · VLA-Cache — https://arxiv.org/abs/2502.02175 · ADP — https://arxiv.org/abs/2509.22093 · VLA-Pruner — https://arxiv.org/abs/2511.16449 · CLP "Fewer Layers Than You Think" — https://arxiv.org/abs/2606.20246 · FastVLM — https://arxiv.org/abs/2412.13303 · ToMe — https://arxiv.org/abs/2210.09461

*Local code anchors: `smolvla/modeling_smolvla.py` — `sample_actions:812` (prefix cached once `:836`; fixed Euler `:845/:876`), `denoise_step:883` (cache re-read `:904`), `embed_image`. `smolvla/smolvlm_with_expert.py` — layer truncation `:102`, dynamic-cache StaticCache TODO `:272/:276`, eager attn interface. `smolvla/configuration_smolvla.py` — `use_cache`, `num_steps`, `num_vlm_layers=16`, `self_attn_every_n_layers=2`, `compile_model`, `rtc_config`. `rtc/configuration_rtc.py`.*