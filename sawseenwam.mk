# Fine-tune SawSeenWAM on LIBERO and evaluate. All targets run inside the
# LIBERO benchmark Docker image so libero/MuJoCo/CUDA are pre-configured.
#
# SawSeenWAM = SawSeenVLAWM with the **new** le-wm: separate per-camera ViT
# encoders (no horizontal concat) and a variable-stride action_encoder
# supporting multi-offset cost methods. The new le-wm lives at
# /home/lucius/data/personal-hive/code/le-wm. Defaults assume the
# corresponding new-format `_object.ckpt`. Registered as
# `--policy.type=sawseenwam`. See design/SawSeenWAM.md (TBD) for the diff.
#
#   make -f sawseenwam.mk build   # build LIBERO benchmark Docker image (same as smolvla.mk / sawseenvlawm.mk)
#   make -f sawseenwam.mk train   # fine-tune in the container
#   make -f sawseenwam.mk eval    # eval the trained policy
#   make -f sawseenwam.mk eval-mpc MPC_HORIZON_MODE=multi_offset MPC_OFFSETS=4,8,16
#
# Override any variable on the command line, e.g.
#   make -f sawseenwam.mk train LEWM_CKPT_NAME=lewm_dual_epoch_10_object.ckpt
#   make -f sawseenwam.mk eval EVAL_POLICY=$(OUTPUT_DIR)/checkpoints/last/pretrained_model

DOCKER_IMAGE   ?= lerobot-benchmark-libero
HF_CACHE_DIR   ?= $(HOME)/.cache/huggingface
LIBERO_CACHE_DIR ?= $(HOME)/.cache/libero
# Host directory containing the **new** le-wm libero checkpoint. By
# default we use a dedicated dir to avoid mixing with the old single-
# encoder checkpoints that sawseenvlawm.mk consumes.
LEWM_HOST_DIR  ?= $(HOME)/.stable-wm/libero_dual
LEWM_CKPT_NAME ?= lewm_dual_epoch_10_object.ckpt
# le-wm source directory — the **new** le-wm repo, required for the
# pickled JEPA to unpickle (its module/jepa classes must be importable).
LEWM_SRC_DIR   ?= $(HOME)/data/personal-hive/code/le-wm
# Host directory where checkpoints / eval results land. Bind-mounted at
# the same absolute path inside the container, so OUTPUT_DIR is a single
# host-equals-container path that ``--output_dir`` consumes directly.
OUTPUTS_HOST_DIR ?= /mnt/hard_disk_16Tb/data/lerobot/outputs
GPU            ?=

# Train
DATASET_REPO   ?= HuggingFaceVLA/libero
OUTPUT_DIR     ?= $(OUTPUTS_HOST_DIR)/train/sawseenwam_libero_12k_bs64_dualenc_lge_2xGPUs_bf16
JOB_NAME       ?= sawseenwam_libero_dualenc
STEPS          ?= 12000

# bs=64 per GPU on 24 GB cards with LATENT_GOAL=true. With the new dual-
# encoder lewm we run two 224×224 ViT-Tiny forwards per step instead of
# one 224×448; per-iter compute is roughly comparable (twice as many
# encoder forwards, half as many tokens each) so the same batch size
# should fit. If OOM, drop to bs=48 first.
BATCH_SIZE     ?= 64
NUM_WORKERS    ?= 4
SAVE_FREQ      ?= 1000
LOG_FREQ       ?= 100

# Sqrt-scaled from the bs64 baseline (LR=4e-4 at global_batch=128):
# LR ≈ 4e-4 × sqrt(global_batch/128). Default tuned for BATCH_SIZE=64
# NUM_GPUS=2 (global_batch=128 → LR=4.0e-4). Re-scale if you change either.
LR             ?= 4.0e-4

COMPILE_MODEL   ?= false
COMPILE_MODE    ?= max-autotune
PAD_LANGUAGE_TO ?= max_length
DEVICE         ?= cuda
WANDB          ?= false
TENSORBOARD    ?= true
NUM_GPUS       ?= 2
MIXED_PRECISION ?= bf16

# ── le-wm dual-encoder side-channel ──────────────────────────────────
# Per-camera ViT input — 224×224 square, not the 224×448 concat the old
# lewm used. Keep at 224 unless your checkpoint was trained at a non-
# standard resolution.
LEWM_NUM_TOKENS  ?= 1
LEWM_FREEZE      ?= true
LEWM_IMAGE_H     ?= 224
LEWM_IMAGE_W     ?= 224

# Ordered tuple of pixel keys feeding into the dual encoders. The first
# entry binds to encoders[0] (agentview), the second to encoders[1]
# (eye-in-hand). Must match the order baked into the loaded JEPA's
# .pixel_keys (the new-lewm CLAUDE.md calls this load-bearing).
LEWM_PIXEL_KEYS  ?= '("pixels","pixels_wrist")'
LEWM_MULTI_TOKEN ?= false

LEWM_INJECT_TO   ?= suffix

# ── Latent Goal Expert (LGE) ─────────────────────────────────────────
LATENT_GOAL              ?= true
LATENT_GOAL_LOSS_WEIGHT  ?= 1.0
LATENT_GOAL_SIGREG_WEIGHT ?= 0.1

PEFT             ?= false
LORA_R           ?= 16

# Mode 3 — feed [z_t, z_g] tokens into the action expert's suffix.
LATENT_GOAL_INJECT_TO_ACTION ?= true
LATENT_GOAL_INJECT_Z_G_SOURCE ?= scheduled
LATENT_GOAL_INJECT_SCHEDULE_START_STEP ?= $(shell expr $(STEPS) / 2)
LATENT_GOAL_INJECT_SCHEDULE_END_STEP ?= $(STEPS)
LATENT_GOAL_INJECT_DETACH ?= true
LATENT_GOAL_TRAIN_NUM_STEPS ?= 10

# LGE target offset — frame at +N from the anchor is the LGE z_g target.
# Decoupled from chunk_size in SawSeenWAM so the LGE supervision aligns
# with the new lewm predictor's max k_tail (25 in the latest checkpoint).
# Set to 0 / empty to fall back to chunk_size (legacy behavior).
LATENT_GOAL_TARGET_OFFSET ?= 25

# ── Phase B / MPC inference (eval-only) ──────────────────────────────
MPC                ?= false
MPC_SCHEME         ?= anchor_perturb  # anchor_perturb | cem | mppi
MPC_NUM_CANDIDATES ?= 16
MPC_NOISE_SCALE    ?= 0.1
MPC_CEM_NUM_ITER   ?= 4
MPC_CEM_TOPK       ?= 4
MPC_CEM_BLEND      ?= 0.5

MPC_CEM_INCLUDE_ANCHOR ?= true
MPC_CEM_INIT_MEAN      ?= anchor   # anchor | zero
MPC_CEM_RETURN         ?= best_ever # best_ever | final_mean

MPC_MPPI_TEMP      ?= 1.0
MPC_MPPI_NUM_ITER  ?= 4
MPC_SCORE_FLOOR_MARGIN ?= 0.15
MPC_ICEM_BETA      ?= 0.0

# ── Varied-horizon MPC (new-le-wm only) ──────────────────────────────
# "single" — AR rollout at k=1 for chunk_size steps, MSE to LGE z_g.
#            Matches SawSeenVLAWM behavior.
# "multi_offset" — for each k in MPC_OFFSETS, single-shot var-stride
#            prediction at k_tail=k from the 3-slot history, weighted
#            sum of MSEs to the same z_g.
MPC_HORIZON_MODE   ?= single
# Comma-separated k_tail offsets. Each must be in the checkpoint's
# trained k_choices. Max value must be ≤ chunk_size (the longest
# offset slot needs that many candidate actions).
# The latest new-lewm checkpoint trains with k_choices=(1,2,5,10,25),
# so MPC_OFFSETS values must be ⊆ that set. (25,) is the largest
# single-offset; mixed e.g. (10,25) gives multi-horizon scoring.
MPC_OFFSETS        ?= '(25,)'
MPC_OFFSET_WEIGHTS ?= '(1.0,)'
# Encoder's k_max — baked into the JEPA at training. Latest new-lewm
# uses k_max=25 (k_choices=(1,2,5,10,25)); set to 16 if loading an
# earlier checkpoint with k_choices=(1,2,4,8,16).
MPC_ACTION_K_MAX   ?= 25

TRAIN_LAUNCHER  = $(if $(filter-out 1,$(NUM_GPUS)),accelerate launch --multi_gpu --num_processes=$(NUM_GPUS) --mixed_precision=$(MIXED_PRECISION) -m lerobot.scripts.lerobot_train,lerobot-train)
DOCKER_CUDA_ENV = $(if $(filter-out 1,$(NUM_GPUS)),-e CUDA_VISIBLE_DEVICES=$(shell python3 -c "print(','.join(str(i) for i in range($(NUM_GPUS))))"),)

# Eval
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
	  --policy.type=sawseenwam \
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
	  --policy.lewm_pixel_keys=$(LEWM_PIXEL_KEYS) \
	  --policy.lewm_multi_token=$(LEWM_MULTI_TOKEN) \
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
	  --policy.latent_goal_target_offset=$(LATENT_GOAL_TARGET_OFFSET) \
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

# Phase B / MPC eval — overlay on top of EVAL_POLICY. Pass
# MPC_HORIZON_MODE=multi_offset to enable the new variable-stride cost.
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
	  --policy.mpc_horizon_mode=$(MPC_HORIZON_MODE) \
	  --policy.mpc_offsets=$(MPC_OFFSETS) \
	  --policy.mpc_offset_weights=$(MPC_OFFSET_WEIGHTS) \
	  --policy.mpc_action_k_max=$(MPC_ACTION_K_MAX) \
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
