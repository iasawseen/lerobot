# SawSeenVLAWM TODO

Forward-looking work for the sawseenvlawm policy. Items are independent
unless explicitly noted; each can land as its own PR.

---

## 1. LoRA for the frozen VLM

**Status:** scoped, not implemented.

The sawseenvlawm policy freezes the entire SmolVLM2 backbone via
`train_expert_only=True`. PEFT is wired in at the lerobot level (see
`src/lerobot/configs/default.py:PeftConfig` and
`SawSeenVLAWMPolicy._get_default_peft_targets`), but the current target
regex was inherited from sawseenvla and only adapts the action expert,
not the VLM.

Add LoRA adapters to the VLM's attention `q_proj`/`v_proj` so the VLM
itself can adapt to libero's distribution while staying parameter-cheap.

**Concrete changes:**

- Update `_get_default_peft_targets` in `modeling_sawseenvlawm.py`:

```python
target_modules = r"model\.vlm_with_expert\.vlm\..*\.self_attn\.(q|v)_proj"
modules_to_save = [
    "lm_expert", "lgp_expert",
    "state_proj",
    "action_in_proj", "action_out_proj",
    "action_time_mlp_in", "action_time_mlp_out",
    "lgp_in_proj", "lgp_out_proj", "lgp_anchor_proj",
    "lgp_time_mlp_in", "lgp_time_mlp_out",
    "lewm_proj",
]
```

- Add `PEFT ?= false` and `LORA_R ?= 16` knobs to `sawseenvlawm.mk`,
  conditionally append `--peft.method_type=LORA --peft.r=$(LORA_R)` to
  the train target.

**Trainable params delta:** ~0.5M (16 layers × 2 projs × 2·960·16 at r=16)
on top of the existing ~196M expert + projection params.

**Open questions:**
- Q/V only vs. Q/K/V/O — Q/V is the LoRA paper default; Q/K/V/O ~2× the
  adapter params and sometimes helps on multimodal tasks.
- Default `r` — 16 (lerobot's `PeftConfig` default), 32, or task-specific?
- LoRA on SigLIP vision encoder too? Currently frozen; probably not
  needed for libero scenes but worth ablating later.

**Scope:** ~10 LOC + makefile.

---

## 2. KI + FAST + LoRA

**Status:** not implemented anywhere in lerobot. New addition.
**Depends on item 1** (LoRA scaffolding for the VLM).

The π0.5 / π0.6 papers train the VLM jointly on **FAST action tokens**
(cross-entropy on a discrete action vocabulary) while keeping the
flow-matching action expert's gradients **insulated** from the VLM
(`prefix_embs.detach()` before the expert reads them). Result: the VLM
learns about actions and adapts to the embodiment, but its language
knowledge isn't degraded by the noisy flow-matching gradient.

The π paper does this with **full-VLM training** along the CE loss.
That's not feasible on our hardware: we're on 2× RTX 3090 (24 GB each),
already at ~21 GB memory pressure with full FT of the experts at
bs=64. Adding ~500M trainable VLM params + Adam state would OOM.

**Use LoRA as the gradient channel for the VLM.** The FAST CE loss
flows into the VLM through LoRA adapters only; full VLM weights stay
frozen. This keeps the KI mechanism intact (the VLM "knows" actions
because its LoRA adapters were trained against FAST token CE) while
keeping the trainable parameter budget manageable on 2× 3090.

Lerobot already has the FAST tokenizer (`lerobot/fast-action-tokenizer`),
used by `pi0_fast` (single-head autoregressive) and `wall_x`
(diffusion-or-fast, mutually exclusive). Neither does the joint KI
scheme. Bringing it to sawseenvlawm with LoRA is new territory.

**Sketch:**

1. **LoRA on VLM** (from item 1). PEFT regex targets
   `model.vlm_with_expert.vlm.*.self_attn.(q|v)_proj`; experts +
   projections in `modules_to_save`. Same scaffolding either way; KI
   is what changes the loss attached to those LoRA paths.
2. **VLM lm_head for FAST vocabulary.** Add a small classifier head on
   top of the VLM's last hidden states that predicts FAST action token
   IDs autoregressively. The head's vocabulary matches
   `lerobot/fast-action-tokenizer`. The head is fully trainable
   (`modules_to_save`).
3. **Pre-tokenize action chunks** via the FAST tokenizer in the
   pre-processor (or on-the-fly in the policy `forward`). Ship discrete
   token IDs in the batch alongside continuous actions.
4. **Cross-entropy loss** on VLM-predicted FAST tokens. Gradients flow
   into the VLM through LoRA adapters (and the lm_head, which is
   `modules_to_save`).
5. **Detach the VLM→expert handoff.** In `VLAFlowMatching.forward`,
   replace `prefix_embs` with `prefix_embs.detach()` before passing to
   `embed_suffix` / the action expert path. The flow-matching loss
   propagates through the experts but stops at the VLM boundary —
   neither the action gradient nor the LGP gradient affects the VLM
   LoRA adapters.
6. **Loss combination:**
   `L = L_action(detached prefix) + λ_lgp · L_lgp(detached prefix) + λ_ki · L_ki(prefix gradients → VLM LoRA)`

**Memory budget on 2× 3090:**
- VLM LoRA adapters at r=16: ~0.5M trainable params, ~8 MB Adam state.
- FAST lm_head (192-class? 1024-class?) on hidden=960: ~1M params.
- Action expert + LGP expert + projections: ~196M (unchanged).
- **Total trainable: ~197M** vs. the ~696M of "full VLM + experts."
  Roughly the same as the current full-FT-experts setup, so should fit
  at the same bs=64 we're already using. KI doesn't blow up the budget.

**Config additions:**

- `ki_enabled: bool = False`
- `ki_loss_weight: float = 1.0`
- `fast_tokenizer_path: str = "lerobot/fast-action-tokenizer"`
- `max_action_tokens: int` (existing in `pi0_fast`; copy the convention)

When `ki_enabled=True`, validation should also require `peft.method_type
== "LORA"` — KI without LoRA is the full-VLM scheme that won't fit.

**Open questions:**
- Bit-exact reproduction of the π paper's gradient stop point — does
  detach happen before or after RMSNorm at the VLM tail?
- Do we use the existing pre-trained FAST tokenizer or train a libero-
  specific one via `lerobot_train_tokenizer.py`?
- Causal vs. parallel decoding for the FAST CE loss? π paper does
  causal; cheaper to compute and matches autoregressive decoding.
- LoRA rank for KI: r=16 is a starting point. If FAST CE doesn't move
  on r=16, try r=32 or r=64 before concluding KI doesn't help.

**Scope:** ~150–200 LOC + config + makefile. Builds on item 1's LoRA
scaffolding.

---

## 3. VLAWM hybrid (Phase B of Future Sight)

**Status:** designed in `design/future-sight-implicit-wm.md`, Phase A
(LGP head training) is implemented and in active testing. Phase B is
the inference-time MPC inner loop and is not implemented.

The full hybrid is: at inference, the LGP gives `z_g` (the goal latent),
the action expert gives an anchor chunk `a*`, the **le-wm JEPA predictor**
rolls forward `chunk_size` steps from `z_t` for K perturbations of `a*`,
and the perturbation that minimizes `d(WM(z_t, a*_k), z_g)` is what
gets executed. This is the K-perturbation MPC scaffolding from the
synthesis doc (lines ~31–82).

**Components needed:**

1. **Load the JEPA predictor**, not just the encoder. The le-wm
   checkpoint contains both; `LeWMVisionEncoder.from_lewm_checkpoint`
   currently extracts only `.encoder`. Add `LeWMPredictor` (or extend
   the encoder wrapper) to expose `.predictor` and a forward that takes
   `(z_t, action_chunk) → ẑ_{t+H}`.
2. **Wire inference-time LGP denoising.** Currently `sample_actions`
   falls back to the parent's single-expert path (LGP silent). Add a
   parallel path that runs FS denoising (10 Euler steps) on the FS
   suffix to produce `z_g`. The same prefix KV cache the action expert
   uses can be reused.
3. **K-perturbation MPC inner loop** in `sample_actions` (or a new
   `select_action_mpc` method):
   - Sample anchor `a*` via standard action-expert denoising.
   - Sample K Gaussian perturbations: `a*_k = a* + ε_k`.
   - Roll WM: `ẑ_k = predictor(z_t, a*_k)` for each k.
   - Score: `s_k = ||ẑ_k − z_g||₂` (or learned norm).
   - Return the first action of `a*_{argmin s_k}`.
4. **Score-floor escape** (synthesis doc, mitigation 2): if
   `min_k s_k > τ`, resample anchor at higher entropy.
5. **Mode toggle** in config: `inference_mode: str = "off" | "mpc"`
   so we can A/B Mode 1 (anchor only) vs. Mode 2 (full MPC) without
   reloading the policy.

**Validation pre-reqs:**
- Phase A must show LGP retrieval-probe accuracy > random chance
  (otherwise the MPC scorer is comparing against noise).
- WM predictor needs verification it accepts arbitrary action chunks
  (training distribution matters; the doc's Step 0 failure-mining is
  what makes the predictor dynamics-complete on perturbations).

**Open questions:**
- Direct L2 vs. learned norm for the scorer — synthesis doc, "Distance
  metric" section. Start with L2; add learned norm as an ablation.
- K (number of perturbations) — doc suggests 8 as a starting point.
- Anchor-only fallback when MPC fails: invariant we want is "Mode 2
  never under-performs Mode 1." If MPC scoring is bad, we should fall
  back, not commit to a poor candidate.
- Distillation (Phase D): once Mode 2 is working, train a Mode 3 student
  (action expert only) on Mode 2's selected actions. Brings MPC quality
  back to action-expert wall-clock cost.

**Scope:** large. This is multiple weeks. Phase B alone is ~2–3 weeks
of implementation + ablations.

---

## 4. Test inverse-square-root LR scheduler

**Status:** not implemented in lerobot. Reference implementation:
[fairseq inverse_square_root_schedule.py](https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py).

The current default scheduler for sawseenvla / sawseenvlawm (and most
flow-matching policies) is `CosineDecayWithWarmupSchedulerConfig`.
Cosine decay requires committing to a `num_decay_steps` up front and
drives the LR to ~zero by the end. The fairseq inverse-sqrt scheduler
is a popular alternative for transformer training (originating from
"Attention Is All You Need"):

```
lr(t) = peak_lr × min(1, t / warmup_steps) × sqrt(warmup_steps / max(t, warmup_steps))
```

- Linear warmup from 0 → `peak_lr` over `warmup_steps`.
- After warmup: `lr = peak_lr × sqrt(warmup_steps / t)` — smooth decay
  that asymptotes rather than hitting zero.

**Why it's worth testing here:**

- **Horizon-agnostic.** No need to set `scheduler_decay_steps`. Useful
  when we extend training step count mid-experiment without re-tuning
  the schedule.
- **Less aggressive late-stage decay.** Cosine drops the LR ~50× by the
  end; inverse-sqrt at `t = 8000, warmup=1000` only drops ~3×. Lets
  late-training continue to make progress, which may matter for
  capacity-bound runs.
- **Lerobot already has the registration scaffolding** —
  `LRSchedulerConfig` in `src/lerobot/optim/schedulers.py` registers
  via `draccus.ChoiceRegistry`. Adding a new variant is a clean
  drop-in.

**Concrete changes:**

- Add `InverseSqrtSchedulerConfig` to
  `src/lerobot/optim/schedulers.py`:

```python
@LRSchedulerConfig.register_subclass("inverse_sqrt")
@dataclass
class InverseSqrtSchedulerConfig(LRSchedulerConfig):
    peak_lr: float
    num_warmup_steps: int

    def build(self, optimizer, num_training_steps):
        warmup = max(1, self.num_warmup_steps)
        decay_factor = self.peak_lr * warmup ** 0.5
        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            return decay_factor / (self.peak_lr * step ** 0.5)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

- Optionally expose via the policy's `get_scheduler_preset` so
  sawseenvlawm can opt in by config (or by CLI override:
  `--scheduler.type=inverse_sqrt --scheduler.peak_lr=4e-4
  --scheduler.num_warmup_steps=1000`).

**Validation:** run two matched 8k-step sawseenvlawm jobs, one cosine
one inverse-sqrt, same seed and effective batch. Compare:
- Final action loss
- LGP retrieval accuracy on a held-out slice
- Eval success rate on libero suites

If inverse-sqrt is within noise of cosine, prefer it for the
horizon-agnostic property. If it's better late in training, consider
making it the default for long runs.

**Open questions:**
- Should we also add `min_lr_ratio` (cap the decay floor)? Fairseq's
  version doesn't, but it can stabilize very long runs.
- Couple `num_warmup_steps` to a fraction of `cfg.steps` like cosine,
  or decouple? The horizon-agnostic property argues for absolute
  warmup steps.

**Scope:** ~30 LOC for the scheduler class + a one-line CLI override
test, plus the comparison run.

---

## Cross-references

- [`design/SawSeenVLAWM.md`](./SawSeenVLAWM.md) — current
  implementation: side-channel (parked) + LGP Phase A (active).
- [`design/future-sight-implicit-wm.md`](./future-sight-implicit-wm.md) —
  synthesis: motivation, four-phase recipe, MPC scaffolding, ablation
  matrix.

## Suggested ordering

1. **Inverse-sqrt LR scheduler** first — smallest diff, orthogonal to
   architecture changes, gives a horizon-agnostic alternative to cosine
   that's a drop-in for any subsequent run.
2. **LoRA for VLM** — small change, gives the VLM a cheap adaptation
   channel and a known-baseline for comparison. Required prerequisite
   for KI+FAST+LoRA below.
3. **Phase B of VLAWM hybrid** — *after* Phase A LGP shows non-trivial
   retrieval accuracy. No point wiring MPC if the FS head isn't
   producing meaningful goal latents.
4. **KI + FAST + LoRA** last — biggest scope, most novel. Builds on
   item 1's LoRA scaffolding (LoRA is the only feasible gradient
   channel for the VLM on 2× 3090). Worth doing once standalone LoRA
   has been characterized so we can attribute any KI uplift correctly.
