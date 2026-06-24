# Improving SmolVLA: Data, Model, Training, Pretraining, Modalities, and World-Modeling Directions

*Design doc — 2026-06-23. Audience: robotics research lead. Target substrate: a ~1B VLA (SmolVLA / SawSeenVLA class) evaluated on LIBERO / RoboCasa-style sim. Scale constraint: prefer ideas demonstrated or plausible at ≈0.4B–3B; down-rank anything that only pays off at 7B+. Anchor metric: **SawSeenVLA 75% LIBERO-spatial (4k steps, bs=96)** — every roadmap item is judged against this.*

---

## 1. Executive summary — highest-leverage moves, ranked

Each line is one lever, ranked by expected leverage-per-effort on *this* repo's substrate. "Author inference" marks where the ranking is my judgment, not a cited result.

1. **Continuous chunked decode is already correct — don't regress to discrete tokens.** OpenVLA-OFT's decomposition shows the +19pt jump comes from *leaving* autoregressive token decoding (76.5→90.2→95.3) [OFT]. SmolVLA's flow head already has this; protect it. *Cheap (validation only).*
2. **Re-plan frequency + chunk length are the largest free knobs.** SmolVLA: every-1/10-step ≈ 80–83% vs every-50-step 51.8%; chunk n=10 (84.0%) > n=50 (80.3%) [SmolVLA]. Repo uses `chunk_size=50` with K=10 — sweep down. *Cheap.*
3. **EMA + grad-clip + the existing warmup/cosine** — reliable few-point small-data win on a noisy flow gradient [DiffPolicy/Consistency]. SmolVLA paper doesn't foreground EMA; add it. *Cheap, ~30 LOC.*
4. **Diversity-first sim data, not volume.** Imitation scaling is a power law in *distinct envs/objects*, not demos-per-setting [Lin 2410.18647]; LIBERO's 50-demo/task regime is already past the per-setting knee. MimicGen/RoboCasa-style asset/scene randomization beats piling demos. *Moderate.*
5. **MI / influence demo filtering** (DemInf, CUPID) — one offline pass, keep the top subset; counters lucky/degenerate replay successes [DemInf 2502.08623]. *Cheap.*
6. **Monocular-depth Ego3D positional prior** (SpatialVLA recipe, ZoeDepth/Depth-Anything-V2) — free, no calibration, sim-ready, the single most copy-able 3D mechanism for an RGB policy [SpatialVLA]. *Moderate, input-side only.*
7. **Quantile action normalization** — robust to outliers/heterogeneous sources, cheap [OFT, action-space-design]. *Cheap.*
8. **Knowledge Insulation (π0.5-KI) / FAST-CE-into-VLM via LoRA** — the repo's own scoped-but-unbuilt KI+FAST+LoRA item; trains discrete-token supervision into the VLM while the flow gradient is detached at the VLM boundary [KI 2505.23705]. *Heavy-ish but high-ceiling for language grounding.*
9. **Port a quasimetric/contrastive action-conditioned distance into the MPC scorer** — the le-wm line converged on this as the structural fix for cost-ranking ⊥ SR; still raw L2 in the SawSeen scorer. *Moderate, highest-ceiling world-model lever.*
10. **Theia/RADIO frozen-encoder swap** — distills depth+semantics+geometry into one small robot-tuned backbone, near-zero architecture change [Theia 2407.20179]. *Moderate.*

Explicitly **down-ranked for this scale**: RoboMonkey-style 7B verifier test-time scaling (verifier dwarfs the policy); full-OXE dumping (RT-1-X-at-35M underfit regime); GR00T-scale synthetic pyramids and DreamGen (1500-L40-class offline bill); pixel-space video MPC in the control loop (seconds/action); tactile/event (not simulated in LIBERO/RoboCasa).

---

## 2. Where SmolVLA / SawSeenVLA sits today

**SmolVLA** (2506.01844): frozen SmolVLM2-500M (SigLIP + connector + SmolLM2), flow-matching action expert, interleaved cross/self-attn, ~100M trainable expert, layer-skipping (first 16 of 32 VLM layers), 64 visual tokens/frame via PixelShuffle, async inference. Paper LIBERO (authors' own Table 2): **Spatial 90 / Object 96 / Goal 92 / Long 71, avg 87.3** — note community reproductions (lerobot #2107, #3287) struggle to hit these; treat as implementation-sensitive.

**This repo's `sawseenvla`**: VLM decoder truncated to 16 layers (hidden 960, frozen); expert = deep-copy scaled by 0.75 → 720-d, 16 layers, ~98M trainable; per-layer shared-K/V cross-attn (half self, half cross); flow matching with Beta(1.5,1.0) timestep, MSE on `u_t = noise − action`, K=10, `chunk_size=50`; images 512×512; bf16, `train_expert_only=True`, `torch.compile + pad_language_to=max_length` (1.35×), 2× RTX 3090, bs=96, sqrt-scaled LR.

**Reference bar: SawSeenVLA = 75% LIBERO-spatial @ 4k/bs=96.** Sibling results: SawSeenWAM v3 71%, Qwen-swap 71%, easyx4-mixed 57% (the "do random" tax). Best MPC variant on the 12k LGE+Mode3+SIGReg checkpoint: AI-CEM N=16 M=4 σ=0.1 = 79.25% avg (+0.75pp). le-wm standalone never moved SR across ~21 arms despite 40–58% lower off-manifold rollout error — the binding constraint is **cost-ranking geometry off-manifold**, not WM accuracy.

The repo's architectural choices already match VLA best practice at this scale (frozen VLM, continuous chunked flow, interleaved CA/SA, state-routing, layer-skip, vision-token compression). The gap to push is **data, action-rep tuning, language grounding, depth priors, and the world-model cost surface** — not the backbone.

---

## 3. Data axis

### 3a. Sim mining & generation

All caveats: every family-1/family-4 empirical result below was validated on **<250M** policies (BC-RNN, BC-Transformer, DP3). The data-engine half transfers cleanly (SARS trajectories are policy-agnostic); the *filtering-at-1B* step is unproven. The only ~1B-class demonstration is DreamGen on RoboCasa with GR00T-N1.

| Technique | What | Evidence | Scale-applicability | Cost | Plug-in to this repo |
|---|---|---|---|---|---|
| **MimicGen** [2310.17596] | Object-centric segment + transform + physics replay; keep successes | 50K demos / 18 tasks from ~200 source; data engine behind RoboCasa | High — embodiment/policy-agnostic SARS | Cheap–moderate (replay throughput) | Generate broad-pose LIBERO/RoboCasa demos, co-train SawSeenVLA |
| **SkillGen/SkillMimicGen** [2410.18907] | Segment into skills + planned transit, stitch | Higher gen-success on long-horizon/contact-rich than MimicGen | High, same SARS | Moderate (planner + segmentation) | LIBERO-Long / RoboCasa composites where blind replay fails |
| **IntervenGen** [2405.01472] | Synthetic corrective interventions near failure states | Up to 39× robustness from ~10 human interventions | High; underused for VLAs | Cheap–moderate (reset-to-state) | Mine recovery data once you log SawSeen runtime failures; targets BC compounding error |
| **DreamGen / GR00T-Dreams** [2505.12705] | Fine-tune I2V WM → neural-trajectory videos → IDM/LAPA pseudo-actions → co-train | RoboCasa 300-traj 49.6→57.6% (**+8pt, co-train only**; neural-only 20.6%); regime-dependent (+5.9pt @30, +7.9pt @100) | Demonstrated at ~2B GR00T-N1 — *the* on-target result | **Expensive** (240K samples = 54h × 1500 L40) | Defer until cheap replay diversity exhausted |
| **DemoGen** [2502.16932] | Spatial aug w/o physics; synthesize matching point cloud | 74.6% / 8 tasks from 1 demo; 22s vs MimicGen 83.7h | Partial — **point-cloud-locked** | Very cheap but modality-locked | Only if SawSeen ever ingests depth/PC |
| **RoboGen** [2311.01455] | LLM proposes tasks/scenes, learns w/ RL/MP/trajopt | Qualitative diversity stream; no SR headline; no sim2real | Indirect (needs distillation) | Moderate–expensive | Diversity source; distill into SawSeen |

**Scaling-law prior** [Lin 2410.18647]: generalization is a power law in distinct envs/objects, *not* demos-per-setting; past ~tens/setting, marginal gain ≈ 0. **Implication for this repo:** at LIBERO's 50-demo/task you are past the knee — MimicGen with *randomized assets/scenes* moves SawSeenVLA more than 50K demos of the same layouts.

**Demo filtering** (cheap, under-exploited):

| Technique | What | Evidence | Cost |
|---|---|---|---|
| **DemInf** [2502.08623] | VAE-structured low-dim rep → estimate state–action MI; keep high-MI | "up to 10% improved success"; *not* k-NN (corrects a common mischaracterization) | Cheap (one feature pass) |
| **CUPID** [2506.19121] | Influence-function curation | More recent, arguably stronger than DemInf | Cheap–moderate |
| **Belkhale/Cui/Sadigh** [2306.02437] | Quality = action divergence × transition diversity; "diversity isn't free" | Framework; some diverse demos *hurt* BC | Conceptual |

Plug-in: rank the easyx4 mined pile (432 expert + 8456 mined) by MI/influence and feed only the top fraction. This directly attacks the **easyx4 "do random" tax (57%, +0.15 loss despite 6.5× data)** — the problem is almost certainly low-MI, label-mismatched mined episodes diluting the gradient, which MI/influence filtering exists to fix. *Author inference, but well-grounded.*

### 3b. Augmentation

The recurring finding: **cheap random background compositing beats heavy diffusion scene synthesis** on scene generalization [GreenAug] — but GreenAug's +36pt was measured on small from-scratch CNN policies, not a frozen-SigLIP 1B VLA; treat the *delta* as off-scale, *direction* as relevant.

| Technique | What | Evidence | Cost | Plug-in |
|---|---|---|---|---|
| **Random 90% crop + mild color jitter** | OpenVLA default: random-90%-area train / center-90% test | CV-aug 70 vs NoAug 55 [GreenAug]; the floor | Near-zero | **Match train/test crop ratio** — critical for frozen SigLIP; aggressive crops push off the pretrained manifold and you can't fine-tune back |
| **LLM instruction paraphrasing** | N paraphrases/instruction, cached offline | Robustness ↑, slight top-1 token ↓ [OpenVLA study 2603.16044, 100-traj] | Very cheap | Frozen VLM makes it low-risk; teaches the *expert* not to overfit phrasing. Watch exact-instruction regression — directly relevant to the INSTRUCTION_LABELS work |
| **Delta actions + mild action/proprio noise** | Target-rep choice + DART-style noise | IL folklore-grade; delta favors short windows [action-space-design] | Trivial | Ablate vs current target rep |
| **Temporal ensembling** | Exp-weighted overlapping chunks | ACT modest +3.3%; fails under latency | Free, inference | Cheap smoothness; superseded by RTC under latency |
| **GreenAug-Rand** | Random-texture background swap | Random > semantic backgrounds | Needs green-screen data | Real-robot only; N/A for LIBERO sim |
| **Viewpoint jitter** | Synthetic camera-pose perturbation | Viewpoint = **#1 VLA fragility** [LIBERO-Plus] | Moderate in sim | Highest-value robustness aug; render multi-view in LIBERO/RoboCasa |
| **Heavy diffusion aug (GenAug/ROSIE/CACTI)** | Inpaint objects/scenes | ~40% gen improvement [GenAug] but beaten by random compositing | Medium-high offline | Skip unless you need object-category diversity |

**Correction to the common LIBERO-Plus framing:** models are **most fragile to camera viewpoint and robot initial state**, then object layout, then sensor noise — and **most resilient to lighting and background** (wrist-camera geometric cues). So lighting jitter is harmless but *not* high-value; **viewpoint + initial-state** (the latter not fixable by any pixel aug — needs proprio/init-state randomization) are the real targets.

**Skip co-training entirely**: its purpose is preventing VLM catastrophic forgetting during action FT. With a **frozen** VLM, priors are preserved by construction — a structural advantage of this recipe. Reallocate that compute to viewpoint aug + the action expert.

---

## 4. Model axis

### 4a. Action head & architecture

| Technique | Effect | Scale | Cost | This repo |
|---|---|---|---|---|
| **Flow-matching expert (baseline)** | Multimodal, smooth, few-step; π0 10 steps / SmolVLA K=10 | Substrate | — | Keep |
| **Parallel continuous decode (OFT)** | 76.5→97.1 LIBERO, 26× throughput; the +19pt is from *leaving AR* | 7B base, ports down | — | Already parallel via flow — protect it |
| **Interleaved CA+SA expert** | 85.5 vs pure-CA 79.0 / pure-SA 74.5 LIBERO [SmolVLA] | 0.45B | Done | Repo's per-layer shared-K/V CA matches this |
| **State → VLM as single token** | SmolVLA routes proprio through the VLM prefix (frozen-VLM regime); a real lever (the specific 80.3-vs-73.3 pair is misattributed in some summaries — it's the layer-skip ablation) | 0.45B | Near-zero | Verify SawSeen state routing; Qwen Tier-1 "state-as-token inside Qwen" is the analogue |
| **Self-conditioning in flow** | Feed prior velocity estimate back; standard generative trick | — | Low | Scoped Qwen Tier-1; untried |
| **FAST discrete tokens** [2501.09747] | 5× faster *training*; but ~750ms/chunk AR inference vs ~100ms flow | 3B | High + wrong latency direction | Only via **KI** (below), never as the inference head |
| **Knowledge Insulation** [2505.23705] | FAST-CE supervises VLM via LoRA; flow gradient detached at VLM boundary → preserves language without losing flow inference | π0.5-class | Med-high | **The repo's scoped-unbuilt KI+FAST+LoRA item** — highest-ceiling language-grounding lever |
| **MiniVLA residual-VQ chunking** [SAIL] | LIBERO-90 82% vs OpenVLA 62% at ~1B (Qwen-2.5-0.5B) | ~1B | Med | Alternative action tokenization; *but* VQ can average multimodal modes — flow is safer; treat as a contrast, not a swap |

### 4b. Efficiency / inference

| Technique | Effect | Scale | Cost | This repo |
|---|---|---|---|---|
| **RTC** [2506.07339] | Inpaint frozen committed actions; training-free, robust >300ms delay where TE collapses | π0/π0.5/SmolVLA | Near-zero, in LeRobot | **Adopt** for any latency deployment — flow head qualifies |
| **A2C2** [2509.23224] | Per-step correction head; +7pt LIBERO-Spatial over RTC under delay | small module | Low | Orthogonal to RTC; small add |
| **Async inference** | SmolVLA 9.7s vs 13.75s (~30% faster) | 0.45B | Near-zero | Deployment-loop change |
| **One-step flow distillation** [OneDP 2410.21257 / OFP 2603.12480] | 1.5→62 Hz (~41×); OFP self-distills (no teacher), >100× | — | Med | Inference-only nicety; do *after* SR is good. Relevant to **Mode-3/Phase-D distillation** (distill MPC-winner actions into the expert for Mode-1 wall-clock) — the repo's unbuilt item |
| **Vision-token compression (PixelShuffle 64 tok)** | Big FLOP cut, minimal acc loss | 0.45B | Low | Repo already at 512×512; **Perceiver compression** is the scoped Qwen Tier-1 untried alternative |
| **VLA-aware quantization (QVLA W4A4)** [2602.03782] | 99.3% retention @ 28.2% memory; projector is the bottleneck (naive bnb degrades) | 7B base | Med | Deployment-only; use VLA-aware, not vanilla bnb |
| **KV-cache reuse across flow steps** | Cache VLM prefix once, reuse across K denoise steps (architecturally sound for prefix/blockwise attn; *not* measured) | — | Low | Author inference — verify the repo's shared-K/V CA already does this |

---

## 5. Training axis

### Cheap wins (Tier 1 — high effect, near-zero effort)

| Lever | Effect | Scale | Cost |
|---|---|---|---|
| **Keep continuous chunked head, not discrete** | +19pt in OFT decomposition | 7B→ports | validation |
| **Tune chunk H∈[8,30] + re-plan often** | n=10 (84.0) > n=50 (80.3); every-50 = 51.8% [SmolVLA] | 0.45B | config |
| **Freeze VLM, train expert** | Core SmolVLA design; biggest small-data regularizer | 0.45B | already done |
| **EMA + grad-clip(1.0) + warmup+cosine** | Smooths noisy flow gradient; reliable few-pt win | std | ~30 LOC |
| **Quantile (1/99-pct) action normalization** | Outlier-robust on heterogeneous/mined data | — | cheap |
| **Inverse-sqrt LR scheduler** | repo-scoped; stabilizes long runs | — | done/scoped |

AdamW (β2=0.95) is universal — no VLA result beats it at this scale. Peak LR rule: 1e-4 frozen-VLM+expert; ~1e-5 if you LoRA/unfreeze the VLM.

### PEFT

| Lever | Effect | Scale | Cost |
|---|---|---|---|
| **Freeze-VLM + train expert** | Near-optimal default at small data | 0.45B | done |
| **LoRA r=8–16 on LM attn proj** | "adapting the LM matters; vision-encoder LoRA buys ~+2pt" [2512.11921, Phi-2, 0.27% params] | ~3B | low (PEFT) |
| **DoRA** | helps only at very low rank; ≈ LoRA at r≥16 | — | not worth it |
| **QLoRA** | 7B-on-1-GPU trick — irrelevant at ~1B | — | skip |

Repo already shipped LoRA variant-A (r=16, text-Q/V) for sawseenvla; Qwen Tier-1 "LoRA on Qwen's 6 full-attn layers" is the natural next.

### Loss objectives & auxiliary losses

| Lever | Effect | Scale | Cost |
|---|---|---|---|
| **L1 ≈ flow ≈ diffusion on success** | 95.3 vs 95.4 [OFT]; flow wins inference; L1 is unimodal | 7B | — |
| **IDM / latent-action aux (LAPA-style)** | manufactures supervision from action-free video; GR00T co-train +4.2–8.8% (the +6.8 is the 300-regime; **figure unverified, treat as approximate**) | 7B LWM (method scale-agnostic) | med |
| **Future-frame / WM aux (DreamVLA-class)** | predictive structure, robustness; uncertain ROI, high compute | survey-level | high |
| **SIGReg** | repo already uses on the 12k LGE+Mode3 checkpoint | — | done |

### Regularization / small-data

Freeze-or-LoRA the VLM (free if frozen) > **VLM2VLA actions-as-language + LoRA** (>85% VQA retention vs OpenVLA collapse [2509.22195], but 7B-scale) > 10–20% replay co-training (redundant with frozen VLM here) > instruction dropout (cheap, mild) > weight decay (default).

### Curriculum / sampling / RL / distillation

| Lever | Effect | Scale | Cost |
|---|---|---|---|
| **Balanced/weighted task sampling** | fixes multi-task imbalance | — | sampler weight |
| **Filtered / success-weighted BC** | cheap if you have success labels; reward-weighted-regression spirit | — | data filter |
| **ResFiT residual off-policy RL on frozen base** | 14→64%, 23→64% in 15–76min; 200× vs PPO; *leaves base untouched* [2509.19301] | 29-DoF humanoid (not ≤1B) | needs rewards+resets |
| **GRAPE/TPO preference** | reuses failed rollouts; large *relative* gains [2411.19309, magnitudes unverified] | 7B | labeling pipeline |
| **One-step flow distillation** | inference latency only; do last | — | med |

**Scale honesty:** VLM2VLA, GRAPE, ResFiT, OFT-decomposition are all 7B / non-≤1B demonstrations. Mechanisms transfer; absolute deltas do not. The only at-scale evidence for ≤1B is SmolVLA itself + the Phi-2 LoRA paper.

---

## 6. Modalities & representations axis

Sim note: LIBERO/RoboCasa render depth and multi-view *for free*, so depth/PC methods are testable in sim — but a clean-sim-PC policy won't transfer to a real RGB-only SO-101 without a depth camera (sim-to-real PC-noise gap). The honest signal for 3D value is generalization splits, not the ~95% RGB ceiling on easy LIBERO suites.

| Technique | Modality cost | Evidence | Sim-applicability | Plug-in |
|---|---|---|---|---|
| **Monocular Ego3D prior (SpatialVLA)** [2501.15830] | **Free** (ZoeDepth/Depth-Anything-V2), no calibration | numbers confounded w/ 3.5B + 1.1M-ep pretrain — use *mechanism* only | Sim-ready | Back-project depth via intrinsics → MLP-embed → **add to SigLIP tokens**; no action-expert surgery. **Do first** |
| **Depth-prediction aux loss** | Free (target only) | DepthVLA-class depth-aware branch [2510.13375, deltas unverified] | Sim-ready | Cheaper than back-projection input; complements the input-side recipe |
| **Theia / RADIO frozen encoder** [2407.20179] | Free, no sensor | distills CLIP+DINOv2+SAM+Depth-Anything+ViT; beats teachers smaller/less-data | Sim-ready | Near-zero arch change; pair with Ego3D |
| **PointVLA injector** [2503.07511] | RGB-D (render in sim) | +4.7% params on frozen 2D base; occluded 56 vs 38%, unseen 62 vs 47% | Sim (rendered PC) | **Adopt the modular-injection *design*** even with monocular pseudo-PC — mirrors the repo's le-wm side-channel pattern |
| **GP3 / RoboVGGT** [2509.15733] | Multi-view RGB, no depth | MetaWorld +11.2, RLBench +22.7, ALOHA +57.5 | Sim if multi-view captured | Heavier (VGGT latency/scale-shift; fine-tune at 224×224) |
| **DP3 / iDP3** [2403.03954 / 2410.10803] | Calibrated RGB-D (iDP3 camera-frame, no calib) | DP3 +24.2% rel @10 demos; iDP3 ~9/10 novel-scene | Sim w/ rendered depth | Best absolute 3D numbers; defer to real-robot phase |
| **BridgeVLA / RVT-2 / 3D-DA** | Calibrated multi-view RGB-D | RLBench 88.2 / 82 / 81.3 (3D-DA multi-view +18.1 abs over Act3D) | Sim possible, heavy | Gold-standard precision; sensor+calib tax |
| **Proprio-rich state + F/T** | proprio free; F/T moderate sensor | "vision = pre-contact, F/T = post-contact" | LIBERO/RoboCasa don't sim tactile | proprio cheap; F/T low payoff on vision benchmarks |
| **Tactile / audio / event** | high sensor cost | low evidence for manipulation | not simulated | **Skip** |

Repo connection: the **le-wm MODEL.md #1 liability is single-CLS vs spatial-token encoder + proprio-in-latent** — exactly what Ego3D/PointVLA/iDP3 fix on the representation side. Wiring spatial tokens + proprio-in-latent into the SawSeen side-channel/LGE space is the most direct modality upgrade.

---

## 7. Pretraining axis

The governing result: **cross-embodiment/web-scale help in proportion to capacity.** RT-1-X (35M) *underfit and lost to single-robot baselines* on data-rich domains; only RT-2-X (55B) absorbed the full mix [RT-X 2310.08864]. A ≤1B model lives where naive full-OXE *hurts*.

**Cheap (small-lab-reachable):**

| Strategy | Why it works ≤1B | Evidence | Cost |
|---|---|---|---|
| **Web/VQA co-training + state-as-tokens** | small VLAs forget worst; cheapest forgetting fix | π0.5: drop web → OOD 94→74% (figure approximate) | low-med — *but* **redundant if VLM frozen** (this repo's case) |
| **Knowledge Insulation** [2505.23705] | gradient-isolate VLM from action expert + discrete-token co-train | π0.5-class; the single most transferable sub-1B idea | med |
| **Latent-action pretraining (LAPA)** [2410.11758] | 30–40× cheaper than action-labeled (272 H100h vs 21.5k A100h); decouples cost from scale | demonstrated at **7B LWM** — method scale-agnostic, result is not ≤1B | med |
| **Diversity-first curation (~100 envs)** | π0.5: generalization saturates ~104 envs; SmolVLA's community-data thesis | π0.5; SmolVLA | curation effort |
| **Curated *near-embodiment* OXE subset** | positive transfer in low-data regime (RT-1-X +50% there) without the underfit-trap dilution | RT-X | RLDS→lerobot conversion |

**Needs big compute — avoid at this scale/budget:** full-OXE dumping (underfit trap); training your own 1M-hour video encoder; GR00T-scale synthetic pyramids (50k H100h); DreamGen generation (1500-L40 class). The *structure* of the GR00T pyramid (broad pretrain → near-embodiment midtrain → small high-quality post-train) transfers; the literal compute does not.

This repo already does the cheap-correct thing: small community-data pretrain + frozen VLM. The unexploited cheap lever is **KI** (scoped, unbuilt) and **diversity-first mining** (partially explored via VoE/easyx4).

---

## 8. World-modeling & alternatives — connecting to le-wm / LGE / MPC

The repo has gone deep here. External evidence anchors what's worth trying next.

**What the field validates that maps to repo components:**

- **JEPA/DINO-WM latent aux head** [V-JEPA-2-AC 2506.09985, DINO-WM 2411.04983]: frozen-encoder + small action-conditioned latent predictor enables zero-shot latent-MPC (V-JEPA-2-AC: 65–80% pick-place, *strong on reach/place, weak on grasp* — soften any "high" robustness claim). **This is exactly the le-wm side-channel.** The repo's finding that le-wm *never moved SR* despite better rollout accuracy is consistent with the field: the bottleneck is the cost surface, not the predictor.
- **TD-MPC2** [2310.16828]: latent MPC (MPPI/CEM) + **TD-learned terminal value** for beyond-horizon return; single 317M agent / 80 tasks. The repo's MPC is anchor-perturb/CEM/MPPI with **raw L2-to-z_g** and no learned value — TD-MPC2 says add a value function.
- **LAPA / DreamGen** [2410.11758 / 2505.12705]: latent-action pretraining + generative data engine — the LGE/multi-k bank lineage.

**What's genuinely new to try in this repo** (drawn from the repo's own open-gaps list, ranked by ceiling):

1. **Quasimetric (MRN/QRL/IQE) or contrastive-SF action-conditioned distance replacing L2-to-z_g in the SawSeen MPC scorer.** The le-wm research *already converged* on this as the structural fix for "cost-ranking ⊥ SR"; it was never ported into the SawSeen scorer. **Highest ceiling** — the repo has proven the L2 surface is the binding constraint (le-wm SR 0/9–0/47 across 21 arms; MPC rescues weak anchors but regresses strong ones because the L2 cost surface is only meaningful near the on-manifold anchor). *Author inference: this is the single most defensible world-model experiment in the backlog.*
2. **Differentiable MPC** (Adam steps on `cost(z_t,a,z_g)` through the frozen predictor) — bypasses the stochastic noise-floor; the repo flags it as highest-ceiling untried MPC lever. Pairs naturally with #1 (a smooth learned distance is differentiable; L2-in-projector is the degenerate case).
3. **Calibration probe + σ-sweep biased low (0.02–0.05)** — `cost_anchor`/`cost_best`/`elite_std`/`anchor_chosen_share`; repo-flagged top-priority, unbuilt. This *diagnoses* whether #1 is even necessary or whether the scorer is just mis-tuned. **Do this before #1** — cheap, gates the multi-anchor work.
4. **Failure-data mining for the WM predictor (Future Sight Step 0)** — the predictor is dynamics-incomplete off the expert band; no failure-mixed le-wm retrain feeds the SawSeen MPC. Field analogue: IntervenGen recovery data, applied to the *world model* not the policy.
5. **Spatial-token le-wm encoder + proprio-in-latent** (le-wm MODEL.md #1 liability) — feeds both #1 (better latent geometry) and §6 (modality).

Down-ranked: pixel-space video MPC (UniPi/AVDC/Cosmos) — latency-prohibitive in the loop; distill to a latent cost instead. RoboMonkey 7B verifier — dwarfs the policy.

---

## 9. Small-VLA landscape (0.4B–3B + references)

LIBERO numbers are implementation-sensitive; treat cross-row as directional.

| Model | Params | Backbone | Action head | LIBERO | Key trick |
|---|---|---|---|---|---|
| **SmolVLA** | ~0.45B | SmolVLM2 (layer-skip) | Flow expert, interleaved CA/SA | **Spatial 90 / avg 87.3** (authors'; community-hard) | Community-data pretrain + layer-skip + async |
| **SawSeenVLA (repo)** | ~0.6B | SmolVLM2-16L trunc, frozen | 720-d flow expert, shared-K/V CA | **Spatial 75** (4k/bs96) | Dual-encoder, LGE, MPC scorer |
| **OpenVLA-OFT** | 7B | Prismatic-7B (LoRA) | Parallel L1 + chunking | avg **97.1** | Leave AR → continuous chunk |
| **π0 / π0-FAST** | ~3.3B | PaliGemma-3B | Flow / FAST-AR | high when FT | FAST DCT tokens; flow expert |
| **π0.5** | ~3B | PaliGemma-3B | FAST+flow hybrid | strong (community-repro) | Knowledge insulation, open-world |
| **GR00T N1 / N1.5** | ~2.2 / ~3B | Eagle-2 (SmolLM2+SigLIP-2 — *shares SmolVLA lineage*) | DiT flow | strong | Data pyramid; **VLM-freeze → GR-1 lang 46.6→93.3%** |
| **MiniVLA** | ~1B | Qwen-2.5-0.5B + OpenVLA ViT | Residual-VQ chunking | LIBERO-90 82% (vs OpenVLA 62%) | VQ chunking = the sub-1B lever |
| **SpatialVLA** | 3.5B | PaliGemma-2 | Adaptive 3-token grids | Spatial 88.2 / avg 78.1 | Ego3D (ZoeDepth, calib-free) |
| **TinyVLA** | 70M–1.4B | Pythia + LLaVA | Diffusion head | reported | No action pretrain; LoRA; beats DP ~21.5% |
| **CogACT** | ~7B | Prismatic-7B | Componentized DiT | strong | Cognition/action separation; +35% SimplerEnv |
| **RDT-1B** | 1.2B | DiT (scratch, T5-XXL) | Diffusion, 64-step | bimanual/CALVIN | Unified physical action space |
| **Octo** | 27/93M | scratch + T5 | Diffusion readout | pre-LIBERO | 800k-OXE modular |

**What the best small VLAs do that this repo does not:**

1. **VLM freezing to preserve language** (GR00T N1.5: 46.6→93.3% GR-1 language) — repo *truncates* the VLM (drops layers), trading language for size. KI is the cheaper recovery path.
2. **Explicit monocular-depth 3D grounding** (SpatialVLA Ego3D) — repo is pure 2D RGB + flow; le-wm side-channel is the closest but it's JEPA-next-frame, not a 3D positional prior.
3. **Learned action tokenization** (MiniVLA residual-VQ, π0-FAST) — repo regresses via flow; VQ chunking is *the* documented sub-1B lever (caveat: averages multimodal modes).
4. **Knowledge insulation / FAST-CE into VLM** (π0.5) — repo's scoped-unbuilt KI+FAST+LoRA.
5. **A learned MPC cost** (TD-MPC2 terminal value; quasimetric distance) — repo MPC still raw L2-in-projector.

---

## 10. Recommended roadmap — prioritized experiments for THIS repo

Anchor for all: **SawSeenVLA 75% LIBERO-spatial @ 4k/bs=96.** Costs are rough (2× RTX 3090).

| # | Experiment | Hypothesis | Cost | Expected signal | Compare against |
|---|---|---|---|---|---|
| **1** | **Chunk/re-plan sweep** — `chunk_size ∈ {10,20,30,50}` × re-plan freq | n=10 / frequent re-plan beats n=50 open-loop (SmolVLA: 84 vs 80.3; every-50=51.8) | ~4 runs × 4k ≈ 1 GPU-day; ~20 LOC | +3–8pt or flat (validates we're not leaving free points) | SawSeenVLA 75% |
| **2** | **EMA + grad-clip + quantile-norm** | Smoother flow gradient + outlier-robust targets → +1–3pt at small data | ~1 GPU-day; ~40 LOC | small reliable bump; lower loss variance | 75% |
| **3** | **MI/influence filter on easyx4 mined set** (DemInf-style) | The 57% "do random" tax is low-MI dilution; top-fraction filtering recovers toward 75% | ~0.5 GPU-day filter + 1 run; ~150 LOC | easyx4 57 → 65–72%; isolates label-quality vs quantity | easyx4-mixed 57%, ref 75% |
| **4** | **MPC calibration probe + σ-sweep low (0.02–0.05)** | The 79.25% MPC ceiling is scorer-mis-tuned, not surface-broken; low-σ + probe diagnoses | ~1 GPU-day eval-only; ~200 LOC | `anchor_chosen_share`, `elite_std` reveal whether multi-anchor/quasimetric is needed; maybe +1–2pp | AI-CEM 79.25% |
| **5** | **Monocular Ego3D prior** (Depth-Anything-V2 → back-proj → add to SigLIP tokens) | Free 3D positional prior lifts spatial/viewpoint generalization | ~2 GPU-days; ~300 LOC | +2–6pt on spatial suite specifically | 75% |
| **6** | **Quasimetric/contrastive distance in MPC scorer** (replace L2-to-z_g) | The le-wm-proven structural fix; cost-ranking ⊥ SR resolves with a learned action-conditioned distance | ~3 GPU-days (train distance + eval); ~400 LOC | the highest-ceiling swing; could move MPC from +0.75pp to materially positive, or confirm surface is the wall | AI-CEM 79.25%, le-wm SR≈0 |
| **7** | **KI + FAST + LoRA** (FAST-CE into VLM via LoRA, flow gradient detached at VLM boundary) | Recovers language grounding lost to layer-truncation without hurting flow inference | ~3 GPU-days; ~800 LOC (repo-scoped) | better instruction-following / goal-suite; neutral on spatial | 75%, Qwen 71% |
| **8** | **Differentiable MPC** (Adam on `cost(z_t,a,z_g)` through frozen predictor) | Bypasses stochastic noise-floor; pairs with #6's smooth learned cost | ~2 GPU-days; ~300 LOC | gated on #4/#6 outcome; +1–3pp if cost is smooth | AI-CEM 79.25% |
| **9** | **Viewpoint + init-state augmentation** (render perturbed views in LIBERO) | Targets the #1/#2 fragilities (LIBERO-Plus) | ~1.5 GPU-days; ~150 LOC | robustness/generalization split gain; modest in-dist | 75% |
| **10** | **Theia/RADIO frozen-encoder swap** (vs current SigLIP path) | Robot-tuned distilled encoder gives depth+geometry "for free" | ~2 GPU-days; ~200 LOC | +1–4pt or neutral (SigLIP already strong) | 75% |
| **11** | **Spatial-token le-wm encoder + proprio-in-latent** (le-wm MODEL.md #1) | Better latent geometry feeds both LGE and MPC #6 | ~3 GPU-days (le-wm side) + rewire | enables #6; better LGE prediction | le-wm arms |
| **12** | **Mode-3/Phase-D distillation** (distill MPC-winner actions into expert) | One-step Mode-1 wall-clock without MPC at inference | ~2 GPU-days; ~250 LOC | match MPC SR at Mode-1 latency | 79.25% MPC vs Mode-1 |

**Suggested order:** 1→2→3 (cheap data/training wins, de-risk the bar) → 4 (diagnose MPC) → 5 (depth prior, independent track) → 6 (the big world-model bet, gated by 4) → 7 (language, independent) → 8/11/12 (follow-ons). Items 1–4 are <1 GPU-week combined and should run first; 6 is the highest-ceiling/highest-risk and #4 tells you whether to commit to it.

---

## 11. References

Real, deduped. `[unverified]` = fact-check could not confirm a specific number; mechanism/paper is real.

**Data — generation/mining/filtering**
- MimicGen — https://arxiv.org/abs/2310.17596
- SkillMimicGen/SkillGen — https://arxiv.org/abs/2410.18907
- DexMimicGen — https://arxiv.org/abs/2410.24185
- IntervenGen — https://arxiv.org/abs/2405.01472
- DemoGen — https://arxiv.org/abs/2502.16932
- Optimus — https://arxiv.org/abs/2305.16309
- RoboCasa — https://arxiv.org/abs/2406.02523
- RoboGen — https://arxiv.org/abs/2311.01455
- DreamGen / GR00T-Dreams — https://arxiv.org/abs/2505.12705
- DemInf — https://arxiv.org/abs/2502.08623
- CUPID (influence-function curation) — https://arxiv.org/abs/2506.19121
- Data Quality in IL (Belkhale/Cui/Sadigh) — https://arxiv.org/abs/2306.02437
- Quality Diversity IL — https://arxiv.org/abs/2410.06151
- VILD — https://arxiv.org/abs/1909.06769
- Data Scaling Laws in IL (Lin et al.) — https://arxiv.org/abs/2410.18647
- Domain randomization — https://arxiv.org/abs/1703.06907 · https://arxiv.org/abs/1808.00177 · https://arxiv.org/abs/1910.07113

**Augmentation**
- GreenAug — https://arxiv.org/abs/2407.07868
- GenAug — https://arxiv.org/abs/2302.06671 (RSS19 p010)
- ROSIE — https://arxiv.org/abs/2302.11550
- ACT — https://arxiv.org/abs/2304.13705
- RTC — https://arxiv.org/abs/2506.07339
- LIBERO-Plus — https://arxiv.org/abs/2510.13626
- OpenVLA paraphrase study — https://arxiv.org/abs/2603.16044 `[unverified at scale; 100-traj]`

**Model / architecture / efficiency**
- π0 — https://arxiv.org/abs/2410.24164
- SmolVLA — https://arxiv.org/abs/2506.01844
- OpenVLA-OFT — https://arxiv.org/abs/2502.19645
- FAST tokenizer — https://arxiv.org/abs/2501.09747
- Knowledge Insulation — https://arxiv.org/abs/2505.23705
- BAKU — https://arxiv.org/abs/2406.07539
- A2C2 — https://arxiv.org/abs/2509.23224
- OneDP — https://arxiv.org/abs/2410.21257
- One-Step Flow Policy — https://arxiv.org/abs/2603.12480
- RoboMonkey — https://arxiv.org/abs/2506.17811
- QVLA — https://arxiv.org/abs/2602.03782 · BitVLA — https://arxiv.org/abs/2506.07530
- MiniVLA — https://ai.stanford.edu/blog/minivla/

**Training**
- OpenVLA — https://arxiv.org/abs/2406.09246
- ResFiT — https://arxiv.org/abs/2509.19301
- GRAPE/TPO — https://arxiv.org/abs/2411.19309 `[magnitudes unverified]`
- VLM2VLA (Actions as Language) — https://arxiv.org/abs/2509.22195
- Accessible Physical AI (Phi-2 LoRA) — https://arxiv.org/abs/2512.11921 `[74/76% deltas unverified]`
- Demystifying Action Space Design — https://arxiv.org/abs/2602.23408 `[exact deltas unverified]`
- Action Chunking + Exploratory Data — https://arxiv.org/abs/2507.09061
- ACG (conditioning dropout) — https://arxiv.org/abs/2510.22201

**Pretraining**
- Open X-Embodiment / RT-X — https://arxiv.org/abs/2310.08864
- Octo — https://arxiv.org/abs/2405.12213
- π0.5 — https://arxiv.org/abs/2504.16054 `[exact OOD %s read-off-figure]`
- LAPA — https://arxiv.org/abs/2410.11758 (7B, method scale-agnostic)
- Moto — https://arxiv.org/abs/2412.04445
- GR00T N1 — https://arxiv.org/abs/2503.14734 `[+4.2–8.8% co-train delta approximate]`
- LAP — https://arxiv.org/abs/2602.10556 (3B+, not ≤1B; real headline ~50%/2×, not the hallucinated 80/25)

**Modalities / 3D / world models**
- SpatialVLA — https://arxiv.org/abs/2501.15830
- DepthVLA — https://arxiv.org/abs/2510.13375 `[RGB-vs-depth deltas unverified]`
- Depth-Anything-V2 — https://depth-anything-v2.github.io/
- DP3 — https://arxiv.org/abs/2403.03954 · iDP3 — https://arxiv.org/abs/2410.10803
- PointVLA — https://arxiv.org/abs/2503.07511
- BridgeVLA — https://arxiv.org/abs/2506.07961
- RVT-2 — https://arxiv.org/abs/2406.08545 · 3D Diffuser Actor — https://arxiv.org/abs/2402.10885
- GP3 — https://arxiv.org/abs/2509.15733 · VGGT-DP — https://arxiv.org/abs/2509.18778
- RoboPoint — https://arxiv.org/abs/2406.10721
- Theia — https://arxiv.org/abs/2407.20179
- V-JEPA 2 / V-JEPA 2-AC — https://arxiv.org/abs/2506.09985 (strong reach/place, weak grasp)
- DINO-WM — https://arxiv.org/abs/2411.04983
- TD-MPC2 — https://arxiv.org/abs/2310.16828 · DreamerV3 — https://arxiv.org/abs/2301.04104 · DayDreamer — https://arxiv.org/abs/2206.14176
- UniPi — https://arxiv.org/abs/2302.00111 · AVDC — https://arxiv.org/abs/2310.08576 · Genie — https://arxiv.org/abs/2402.15391

**Dropped as hallucinated/anachronistic** (do not cite): RoboCasa365 "arXiv 2603.04356"; any single "value-of-information / VoE surprise-mining demo-selection for VLAs" paper (no canonical source — closest real work is DemInf/IntervenGen/QD-IL); LAP "80%/25% five-video" figure (fabricated; real is ~50%/2× at 3B).

*Cross-cutting scale caveat: of every positive result cited, only SmolVLA and the Phi-2 LoRA paper are demonstrated at ≤1B. RT-1-X (35M) is the cited negative result. OFT/VLM2VLA/GRAPE/ResFiT/RoboMonkey are 7B; LAPA/LAP/SpatialVLA/GR00T/DreamGen are ≥2B. Treat all mechanisms as directional and re-validate absolute deltas at this scale.*