# Fine-tune SawSeenVLAWM on LIBERO and evaluate. All targets run inside the
# LIBERO benchmark Docker image so libero/MuJoCo/CUDA are pre-configured.
#
# SawSeenVLAWM = SawSeenVLA + a frozen le-wm ViT-Tiny encoder feeding the
# action expert via a suffix side-channel. Registered separately as
# `--policy.type=sawseenvlawm` so its checkpoints don't collide with vanilla
# SawSeenVLA. See design/SawSeenVLAWM.md for the architecture.
#
#   make -f sawseenvlawm.mk build   # build LIBERO benchmark Docker image (same as smolvla.mk / sawseenvla.mk)
#   make -f sawseenvlawm.mk train   # fine-tune in the container
#   make -f sawseenvlawm.mk eval    # eval the trained policy in the container
#
# Override any variable on the command line, e.g.
#   make -f sawseenvlawm.mk train STEPS=80000 BATCH_SIZE=8
#   make -f sawseenvlawm.mk train LEWM_NUM_TOKENS=1   # CLS-only ablation
#   make -f sawseenvlawm.mk eval EVAL_POLICY=$(OUTPUT_DIR)/checkpoints/last/pretrained_model

DOCKER_IMAGE   ?= lerobot-benchmark-libero
HF_CACHE_DIR   ?= $(HOME)/.cache/huggingface
LIBERO_CACHE_DIR ?= $(HOME)/.cache/libero
# Host directory containing the le-wm libero checkpoint (mounted into the
# container at /lewm so the policy can find it regardless of host paths).
LEWM_HOST_DIR  ?= $(HOME)/.stable-wm/libero
LEWM_CKPT_NAME ?= lewm_epoch_10_object.ckpt
# le-wm source directory — required because the checkpoint is a pickled
# `JEPA` object that imports `module` and `jepa` modules during torch.load.
# Mounted read-only at /lewm-src and prepended to PYTHONPATH.
LEWM_SRC_DIR   ?= $(HOME)/data/reps/le-wm
# Host directory where checkpoints / eval results land. Bind-mounted at
# the same absolute path inside the container, so OUTPUT_DIR is a single
# host-equals-container path that ``--output_dir`` consumes directly.
# Pre-create the directory if you hit permission errors (docker will
# create it as root otherwise).
OUTPUTS_HOST_DIR ?= /mnt/hard_disk_16Tb/data/lerobot/outputs
GPU            ?=

# Train
DATASET_REPO   ?= HuggingFaceVLA/libero
OUTPUT_DIR     ?= $(OUTPUTS_HOST_DIR)/train/sawseenvlawm_libero_12k_bs64_lewm_proj_lge_sigreg_scheduled_middle_k10_2xGPUs_bf16
JOB_NAME       ?= sawseenvlawm_libero_latent_goal
STEPS          ?= 12000

# bs=64 per GPU on 24 GB cards with LATENT_GOAL=true: the LATENT_GOAL expert adds ~98M
# trainable params + ~2 extra tokens through the full VLM/expert stack on
# top of the side-channel suffix. Drop bs further (32) if you also push
# LEWM_NUM_TOKENS up. Without LATENT_GOAL the side-channel-only configuration
# fits at bs=96 (21 GB at lewm_num_tokens=1, suffix length 50 + 1 = 51).
BATCH_SIZE     ?= 64
NUM_WORKERS    ?= 4
SAVE_FREQ      ?= 1000
LOG_FREQ       ?= 100

# Sqrt-scaled from the bs64 baseline (LR=4e-4 at global_batch=128):
# LR ≈ 4e-4 × sqrt(global_batch/128). Default tuned for BATCH_SIZE=64
# NUM_GPUS=2 (global_batch=128 → LR=4.0e-4). Re-scale if you change either.
LR             ?= 4.0e-4

# Default OFF for the WM variant: torch.compile masks shape mismatches and
# adds a long warmup that's unhelpful while iterating on the lewm wiring.
# Flip to true for production / throughput-sensitive runs once stable.
COMPILE_MODEL   ?= false
COMPILE_MODE    ?= max-autotune
PAD_LANGUAGE_TO ?= max_length
DEVICE         ?= cuda
WANDB          ?= false
TENSORBOARD    ?= true
NUM_GPUS       ?= 2
MIXED_PRECISION ?= bf16

# le-wm side-channel knobs.
# Cameras are concatenated horizontally and fed to the encoder as a single
# image (matches le-wm's libero training distribution: 256x512 raw → 224x448
# after Resize(224)). LEWM_NUM_TOKENS is sliced from the ViT output:
#   1   = CLS-only (cheapest)
#   513 = full grid for 224x448 (16x32 patches + CLS)
LEWM_NUM_TOKENS  ?= 1
LEWM_FREEZE      ?= true
LEWM_IMAGE_H     ?= 224
LEWM_IMAGE_W     ?= 448

# Where lewm tokens enter the model:
#   suffix → projected to expert_hidden_size, prepended to action expert
#   none   → encoder loaded but not injected into the action expert; used for
#            LATENT_GOAL-only ablations where the Latent Goal Expert is the
#            sole le-wm pathway
LEWM_INJECT_TO   ?= suffix

# Latent Goal Expert (LATENT_GOAL) — implementation of the "Future Sight" expert
# from design/future-sight-implicit-wm.md (Phase A). Adds a second
# flow-matching head next to the action expert that regresses to the
# encoded chunk-end frame z_{t+chunk_size} in le-wm's 192-dim latent. The
# active default for sawseenvlawm runs the *combined* configuration —
# le-wm side-channel into the action expert (LEWM_INJECT_TO=suffix) AND
# LATENT_GOAL enabled — to test whether the two pathways stack. Set LATENT_GOAL=false for
# a side-channel-only run, or LEWM_INJECT_TO=none LATENT_GOAL=true for an LATENT_GOAL-only
# ablation (which isolates the FS effect; see design/SawSeenVLAWM.md).
LATENT_GOAL              ?= true
LATENT_GOAL_LOSS_WEIGHT  ?= 1.0

# SIGReg on the LGE's reconstructed clean prediction. Off by default.
# Le-wm uses weight=0.09 for SIGReg on its encoder during JEPA training;
# 0.09 is a reasonable starting point here as well.
LATENT_GOAL_SIGREG_WEIGHT ?= 0.1

# PEFT (LoRA) — when true, LoRA adapters land on the frozen SmolVLM2
# text_model q/v_proj. The action expert, the Latent Goal Expert (when
# on), and all small projections stay fully trainable via
# ``modules_to_save``. Both experts' losses still update LoRA via the
# autograd-connected K/V cache. Off by default since the LGE / Mode 3
# ablations we're benchmarking are full-FT; flip to true for a
# LoRA + LGE ablation.
PEFT             ?= false
LORA_R           ?= 16

# Mode 3 — feed [z_t, z_g] tokens from the Latent Goal Expert into the
# action expert's suffix. Switches training to a sequential 3-pass
# forward (prefix → LGE → action with cached VLM K/V); inference adds a
# K-step LGE denoising pass before the action denoising loop. Off by
# default — flip to true for the LGE-conditioned-action ablation. Pair
# with LEWM_INJECT_TO=none to isolate the LGE pathway.
LATENT_GOAL_INJECT_TO_ACTION ?= true

# Source of z_g going into the action expert during *training*:
#   "encoded"   — frozen le-wm CLS of the chunk-end frame from the dataset.
#                 Train-only; eval still uses LGE's denoised prediction.
#   "predicted" — LGE's clean prediction reconstructed from its velocity
#                 (z_g_pred = x_t - t · v). Matches inference distribution.
#   "scheduled" — per-sample Bernoulli ramp from encoded (teacher) at
#                 step 0 to predicted (student) at LATENT_GOAL_INJECT_SCHEDULE_END_STEP.
#                 Closes the train/eval gap gradually.
LATENT_GOAL_INJECT_Z_G_SOURCE ?= scheduled

# Step at which the "scheduled" source *starts* the linear ramp (p=0
# before this, then ramps from 0 → 1 by LATENT_GOAL_INJECT_SCHEDULE_END_STEP).
# Default 0 = ramp from the very beginning of training. Set > 0 to keep
# the action expert on the clean encoded teacher signal until LGE has
# had time to fit, then start blending in predicted z_g. Must be
# < LATENT_GOAL_INJECT_SCHEDULE_END_STEP. Ignored unless source=scheduled.
LATENT_GOAL_INJECT_SCHEDULE_START_STEP ?= $(shell expr $(STEPS) / 2)

# Step at which the "scheduled" source reaches 100% predicted (linear
# ramp from LATENT_GOAL_INJECT_SCHEDULE_START_STEP). Default = STEPS so
# the schedule completes exactly at the end of training. Ignored unless
# source=scheduled.
LATENT_GOAL_INJECT_SCHEDULE_END_STEP ?= $(STEPS)

# Detach z_g (and z_t) before the action expert reads them. True =
# paper-faithful KI-style barrier (action loss cannot reshape LGE).
# False = differentiable conditioning (LGE also adapts to action loss).
LATENT_GOAL_INJECT_DETACH ?= true

# LGE denoising steps used to produce z_g for the action expert during
# training (Mode 3, source=predicted). 1 = current one-step closed-form
# reconstruction. Set to LATENT_GOAL_NUM_STEPS (10 by default) to match
# the eval inference loop exactly — same train/eval z_g distribution at
# ~2-2.5× per-step wall time. Ignored when source=encoded.
LATENT_GOAL_TRAIN_NUM_STEPS ?= 10

# ── Phase B / MPC inference (eval-only) ──────────────────────────────

# Activates the le-wm predictor rollout + LGE-z_g scoring on top of the
# policy's anchor chunk at inference. All values are runtime knobs read
# by the eval-mpc target; never used by train.
MPC                ?= false
MPC_SCHEME         ?= anchor_perturb  # anchor_perturb | cem | mppi
MPC_NUM_CANDIDATES ?= 16
MPC_NOISE_SCALE    ?= 0.1
MPC_CEM_NUM_ITER   ?= 4
MPC_CEM_TOPK       ?= 4
MPC_CEM_BLEND      ?= 0.5

# CEM variant knobs (scheme=cem only). Mirror le-wm's reference
# CEMSolver semantics. Defaults are the AI-CEM variant (anchor as slot
# 0 every iter, μ_0 = anchor, return best-ever) — preserves prior
# behavior. Set INCLUDE_ANCHOR=false + INIT_MEAN=zero + RETURN=final_mean
# for the pure le-wm reference. Each knob is orthogonal — pick any combo.
MPC_CEM_INCLUDE_ANCHOR ?= true
MPC_CEM_INIT_MEAN      ?= anchor   # anchor | zero
MPC_CEM_RETURN         ?= best_ever # best_ever | final_mean

# MPPI knobs (scheme=mppi only). Temperature β controls softmax
# sharpness: β→0 = hard argmin, β→∞ = uniform mean. Tune against the
# cost scale (sum-of-squares L2 in 192-d projector space).
MPC_MPPI_TEMP      ?= 1.0
# Default M=4 matches CEM's iteration count and total predictor rollouts
# (N·M=64). MPPI-vs-CEM at fixed compute. M=1 = vanilla single-shot.
MPC_MPPI_NUM_ITER  ?= 4
# Score-floor escape (all schemes). Return anchor unchanged unless the
# chosen candidate's predictor cost beats anchor by this relative margin.
# 0.0 = off (legacy behavior); 0.05 = require ≥ 5% relative improvement
# before deviating. Caps MPC's downside on near-perfect anchors at zero
# loss — addresses the strong-anchor regression failure mode.
MPC_SCORE_FLOOR_MARGIN ?= 0.15

# iCEM colored-noise exponent for action-chunk perturbations (all
# schemes). β=0 (default) = white Gaussian noise, legacy behavior.
# β=1 = pink, mild temporal correlation. β=2 = red/Brownian, the
# iCEM default (Pinneri et al. 2020) — slow drifts that keep
# candidates closer to the manifold of real trajectories the le-wm
# predictor was trained on.
MPC_ICEM_BETA      ?= 2.0

TRAIN_LAUNCHER  = $(if $(filter-out 1,$(NUM_GPUS)),accelerate launch --multi_gpu --num_processes=$(NUM_GPUS) --mixed_precision=$(MIXED_PRECISION) -m lerobot.scripts.lerobot_train,lerobot-train)
DOCKER_CUDA_ENV = $(if $(filter-out 1,$(NUM_GPUS)),-e CUDA_VISIBLE_DEVICES=$(shell python3 -c "print(','.join(str(i) for i in range($(NUM_GPUS))))"),)

# Eval
# EVAL_POLICY    ?= outputs/train/sawseenvlawm_libero_32k_bs24_2xGPUs_bf16/checkpoints/last/pretrained_model
EVAL_TASKS     ?= libero_spatial,libero_object,libero_goal,libero_10
EVAL_EPISODES  ?= 10
EVAL_BATCH     ?= 10
EVAL_PARALLEL  ?= 1
EVAL_N_ACTION_STEPS ?= 10

DOCKER_RUN = docker run $(if $(GPU),--gpus device=$(GPU) -e MUJOCO_EGL_DEVICE_ID=0,--gpus all) --rm \
	  --shm-size=8g \
	  -v $(HF_CACHE_DIR):/home/user_lerobot/.cache/huggingface \
	  -v $(LIBERO_CACHE_DIR):/home/user_lerobot/.cache/libero \
	  -v $(LEWM_HOST_DIR):/lewm:ro \
	  -v $(LEWM_SRC_DIR):/lewm-src:ro \
	  -v $(OUTPUTS_HOST_DIR):$(OUTPUTS_HOST_DIR) \
	  -v $(CURDIR)/src:/lerobot/src \
	  -e MUJOCO_GL=egl \
	  -e HF_DATASETS_CACHE=/tmp/hf-datasets \
	  -e WANDB_API_KEY=$(WANDB_API_KEY) \
	  -e ACCELERATE_MIXED_PRECISION=$(MIXED_PRECISION) \
	  -e PYTHONPATH=/lewm-src:/lerobot/src \
	  $(DOCKER_CUDA_ENV) \
	  -w /lerobot \
	  $(DOCKER_IMAGE)

.PHONY: build train eval eval-mpc table

build:
	docker build -f docker/Dockerfile.benchmark.libero -t $(DOCKER_IMAGE) .

train:
	$(DOCKER_RUN) $(TRAIN_LAUNCHER) \
	  --policy.type=sawseenvlawm \
	  --policy.load_vlm_weights=true \
	  --policy.push_to_hub=false \
	  --policy.device=$(DEVICE) \
	  --policy.optimizer_lr=$(LR) \
	  --policy.scheduler_decay_steps=$(STEPS) \
	  --policy.compile_model=$(COMPILE_MODEL) \
	  --policy.compile_mode=$(COMPILE_MODE) \
	  --policy.pad_language_to=$(PAD_LANGUAGE_TO) \
	  --policy.lewm_encoder_path=/lewm/$(LEWM_CKPT_NAME) \
	  --policy.lewm_freeze=$(LEWM_FREEZE) \
	  --policy.lewm_num_tokens=$(LEWM_NUM_TOKENS) \
	  --policy.lewm_image_height=$(LEWM_IMAGE_H) \
	  --policy.lewm_image_width=$(LEWM_IMAGE_W) \
	  --policy.lewm_inject_to=$(LEWM_INJECT_TO) \
	  --policy.latent_goal_enabled=$(LATENT_GOAL) \
	  --policy.latent_goal_loss_weight=$(LATENT_GOAL_LOSS_WEIGHT) \
	  --policy.latent_goal_sigreg_weight=$(LATENT_GOAL_SIGREG_WEIGHT) \
	  --policy.latent_goal_inject_to_action=$(LATENT_GOAL_INJECT_TO_ACTION) \
	  --policy.latent_goal_inject_z_g_source=$(LATENT_GOAL_INJECT_Z_G_SOURCE) \
	  --policy.latent_goal_inject_schedule_start_step=$(LATENT_GOAL_INJECT_SCHEDULE_START_STEP) \
	  --policy.latent_goal_inject_schedule_end_step=$(LATENT_GOAL_INJECT_SCHEDULE_END_STEP) \
	  --policy.latent_goal_inject_detach=$(LATENT_GOAL_INJECT_DETACH) \
	  --policy.latent_goal_train_num_steps=$(LATENT_GOAL_TRAIN_NUM_STEPS) \
	  --dataset.repo_id=$(DATASET_REPO) \
	  --output_dir=$(OUTPUT_DIR) \
	  --job_name=$(JOB_NAME) \
	  --steps=$(STEPS) \
	  --batch_size=$(BATCH_SIZE) \
	  --num_workers=$(NUM_WORKERS) \
	  --save_freq=$(SAVE_FREQ) \
	  --log_freq=$(LOG_FREQ) \
	  --eval_freq=$(STEPS) \
	  --wandb.enable=$(WANDB) \
	  --tensorboard.enable=$(TENSORBOARD) \
	  $(if $(filter true,$(PEFT)),--peft.method_type=LORA --peft.r=$(LORA_R),)

eval:
	$(DOCKER_RUN) lerobot-eval \
	  --policy.path=$(EVAL_POLICY) \
	  --policy.device=$(DEVICE) \
	  --policy.n_action_steps=$(EVAL_N_ACTION_STEPS) \
	  --policy.compile_model=false \
	  --env.type=libero \
	  --env.task=$(EVAL_TASKS) \
	  --eval.n_episodes=$(EVAL_EPISODES) \
	  --eval.batch_size=$(EVAL_BATCH) \
	  --env.max_parallel_tasks=$(EVAL_PARALLEL)

# Phase B / MPC eval: same as eval, plus the MPC inference path. Reuses
# the existing checkpoint at $(EVAL_POLICY); MPC is a runtime overlay.
eval-mpc:
	$(DOCKER_RUN) lerobot-eval \
	  --policy.path=$(EVAL_POLICY) \
	  --policy.device=$(DEVICE) \
	  --policy.n_action_steps=$(EVAL_N_ACTION_STEPS) \
	  --policy.compile_model=false \
	  --policy.mpc_enabled=true \
	  --policy.mpc_scheme=$(MPC_SCHEME) \
	  --policy.mpc_num_candidates=$(MPC_NUM_CANDIDATES) \
	  --policy.mpc_noise_scale=$(MPC_NOISE_SCALE) \
	  --policy.mpc_cem_num_iter=$(MPC_CEM_NUM_ITER) \
	  --policy.mpc_cem_topk=$(MPC_CEM_TOPK) \
	  --policy.mpc_cem_anchor_blend=$(MPC_CEM_BLEND) \
	  --policy.mpc_cem_include_anchor=$(MPC_CEM_INCLUDE_ANCHOR) \
	  --policy.mpc_cem_init_mean=$(MPC_CEM_INIT_MEAN) \
	  --policy.mpc_cem_return=$(MPC_CEM_RETURN) \
	  --policy.mpc_mppi_temperature=$(MPC_MPPI_TEMP) \
	  --policy.mpc_mppi_num_iter=$(MPC_MPPI_NUM_ITER) \
	  --policy.mpc_score_floor_margin=$(MPC_SCORE_FLOOR_MARGIN) \
	  --policy.mpc_icem_beta=$(MPC_ICEM_BETA) \
	  --policy.mpc_predictor_path=/lewm/$(LEWM_CKPT_NAME) \
	  --env.type=libero \
	  --env.task=$(EVAL_TASKS) \
	  --eval.n_episodes=$(EVAL_EPISODES) \
	  --eval.batch_size=$(EVAL_BATCH) \
	  --env.max_parallel_tasks=$(EVAL_PARALLEL)

TABLE_RUN      ?= $(shell ls -td $(OUTPUTS_HOST_DIR)/eval/*/* 2>/dev/null | head -1)
TABLE_LABEL    ?= Policy

table:
	@python3 eval_table.py $(TABLE_RUN) --label "$(TABLE_LABEL)" $(if $(TABLE_LATEX),--latex)
