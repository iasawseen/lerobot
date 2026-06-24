# Instruction labels for mixed (expert + mined) training data

When training a VLA on a dataset that mixes real-task demos with exploration
trajectories ("mined" / no-task episodes), the label assigned to the
non-task episodes is a design lever, not a fixed quantity. This doc lays out
the families, their semantics, and the experimental order to try them in.

## Context

The first run on `libero_spatial_easyx4_smolvla` uses:

- 432 expert episodes with the 10 real LIBERO spatial instructions
- 8456 mined episodes labelled `"do random"`
- Frame split: 15% expert / 85% mined → ~14 expert frames / ~82 mined frames per bs=96 batch

The `"do random"` choice is inherited from
`le-wm/scripts/data/h5_to_lerobot.py:--random-instruction` and is the
simplest possible label: a single English phrase pointing at the
no-task bucket. It's a baseline, not a settled answer.

The deeper question every alternative answers: **what should the
mined examples teach the policy?**

## Five families, in order of increasing semantic richness

### 1. Mode-switch markers (single bucket)

Treat mined data as a single "no-task" mode. The label is a constant.

| label | mechanism | trade-off |
|---|---|---|
| `"do random"` (current) | natural-language phrase | ambiguous: "random" is a word the LM already has priors for |
| `""` (empty) | empty task slot | model learns "no instruction → no-task behavior"; risks blurring real instructions toward the empty mean |
| `"no task"` / `"(none)"` | explicit no-task phrase | similar to "do random", less semantic baggage |
| `<EXPLORE>` (special token) | reserved control token added to vocab | cleanest: one embedding learns to mean "ignore semantic prior", no leakage from English meaning |

**When to use:** when the goal is purely instruction-grounding
regularization (force the model to attend to language). The exact
string doesn't matter once the model has learned the routing — only
*consistency* within the bucket. A reserved token like `<EXPLORE>` is
the cleanest because Qwen's tokenizer would otherwise split "do
random" into 2-3 token pieces that overlap with real English semantics.

**Cost:** zero (string change) up to ~1h (vocab expansion for special
token).

### 2. Behavior-descriptive labels (kinematic anchoring)

Replace `"do random"` with a description of what the mined episode
*actually does* kinematically:

- `"move the gripper randomly"`
- `"explore the workspace"`
- `"freely operate the arm"`
- Per-episode templates from action statistics:
  `"move toward {direction}"` where `direction ∈ {up, down, left, right, ...}`
  computed from net end-effector displacement.

Gives the model a semantic anchor for the action distribution. By
contrast, the model also learns that "pick up X" instructions mean
*something else*, sharpening the language-grounding signal.

**Cost:** 1-2h to wire per-episode templated labels with simple
displacement / gripper statistics.

### 3. VLM-generated pseudo-captions (per-episode unique)

For each mined episode, feed start + end (and maybe mid) frames to a
VLM offline and ask "describe what the robot did". Use that as the
per-episode instruction.

Turns the dataset into a multi-task pretraining set with synthetic but
semantically diverse labels. Instead of 85% gradient signal pointing at
one bucket, we get 85% gradient signal spread across thousands of
varied descriptions ("the robot moved the gripper up and to the left
without grasping anything", etc.).

**Cost:** ~6 GPU-hours of offline VLM inference at 8456 episodes × 2-3
frames. Qwen3-VL or similar is already in the stack. Re-convert the
dataset after caption generation.

**Why this is the highest-leverage option:** transforms the imbalance
*problem* (85% noise) into a *curriculum* (85% diverse instructions).
The expert demonstrations no longer drown in noise — they're one
distribution among many.

### 4. Hindsight goal labels

For each mined episode, label it by its *final state*:

- text: `"achieve gripper-pose {final_xyz}"` or `"reach the position
  where the {object} is at {final_pose}"`
- visual: append final RGB frame as a "goal" conditioning input
  alongside the current frame

This is **goal-conditioned imitation learning** (HER lineage). The
policy implicitly learns to do whatever achieves the labelled final
state.

**Cost:** 3-4h if going text-only with templated phrases. More if
adding a goal-image channel (requires model changes).

**When to use:** when you also want a goal-conditioned policy at
inference time, not just instruction-following.

### 5. Mixed strategy (curriculum)

Combine: 50% mode-switch + 50% behavior-descriptive, or label a
subset with VLM captions and leave the rest as `<EXPLORE>`. Diversifies
the no-task bucket without committing to a single approach.

## Comparison table

| approach | label diversity | research value | offline effort | training effect |
|---|:---:|:---:|:---:|---|
| `"do random"` (current) | 1 | baseline | 0 | mode marker, possible English-prior leakage |
| `<EXPLORE>` token | 1 | medium | 1h + vocab edit | clean mode marker, no language leakage |
| `""` empty | 1 | low | 30 min | risks averaging with real instructions |
| Behavior-templated | ~10-50 | medium-high | 2h | kinematic anchoring |
| **VLM pseudo-captions** | **~8000** | **high** | **6h** | turns noise into curriculum |
| Hindsight goal labels | per-episode | high (different obj.) | 4h | enables goal-conditioned eval |
| Mixed (#2 + #5 + #3) | mixed | high | 3-7h | hedges |

## When to use which — decision flow

```
Is easyx4 eval comparable to spatial-only baseline (within 3pp)?
├── Yes — "do random" already adequate.
│   └── Move on to other variables; revisit only if you want a
│       goal-conditioned policy (→ #4)
└── No — random-data noise is hurting more than helping.
    ├── Try #1 (<EXPLORE> token) first — cheapest test of whether the
    │   issue is label *content* vs *imbalance*.
    └── If #1 doesn't recover, go straight to #3 (VLM pseudo-captions)
        — this is the most powerful transformation of the noise into
        signal and is the de-facto right answer for large mined sets.
```

## Sampling axis (orthogonal to labels)

Independent of *what* the label says, the **sampling weight** between
expert and mined examples also matters. LeRobotDataset defaults to
uniform per-frame sampling, which gives the 15:85 expert:mined split
the dataset naturally has.

- `WeightedRandomSampler` to force 50:50 expert:mined per batch
- Per-epoch curriculum (start expert-heavy, gradually mix in mined)

These compose with any label strategy. Worth ablating once the label
choice is settled.

## What we expect to learn from easyx4 v1

The `easyx4 do random` run is currently in flight (started 2026-06-23,
8k × bs=96 × lr=3.46e-4). At step 1k it's at loss 0.79 vs the
spatial-only reference 0.64 — i.e. **+0.15 above** despite 6.5× more
data. That's the "do random" tax.

If the final spatial eval comes in:
- **≥ 75%** (matches reference): label choice is fine, problem is
  elsewhere.
- **70-75%**: small tax. Worth trying `<EXPLORE>` token (cheap).
- **< 70%**: noise is genuinely hurting. Go to VLM pseudo-captions.

## Open questions

- Can a single Qwen3-VL caption generation pass produce stable
  pseudo-captions across the whole 8456-episode set, or does it need
  temperature=0 + retry to remove jitter?
- For VLM captions, what's the right granularity — one caption per
  episode (cheap) vs one per chunk-step (expensive but matches the
  per-frame instruction granularity LeRobotDataset uses)?
- Does a `<EXPLORE>` token interfere with Qwen's chat template (which
  expects user/assistant turns)? Likely needs to be a free-form text
  with the token inline.

## Related design docs

- [`SawSeenVLAWM.md`](./SawSeenVLAWM.md) — the policy that consumes
  this dataset (will inherit any label-choice decision)
- [`SAWSEENVLA_QWEN.md`](./SAWSEENVLA_QWEN.md) — alternative VLM
  backbone, orthogonal to label choice; whichever wins on spatial
  becomes the substrate for the chosen label strategy
