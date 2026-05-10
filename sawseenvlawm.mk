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
GPU            ?=

# Train
DATASET_REPO   ?= HuggingFaceVLA/libero
# OUTPUT_DIR     ?= outputs/train/sawseenvlawm_libero_10k_bs64_lewm1_lgp_2xGPUs_bf16
OUTPUT_DIR     ?= outputs/train/sawseenvlawm_test
JOB_NAME       ?= sawseenvlawm_libero_lgp
STEPS          ?= 10000
# bs=64 per GPU on 24 GB cards with LGP=true: the LGP expert adds ~98M
# trainable params + ~2 extra tokens through the full VLM/expert stack on
# top of the side-channel suffix. Drop bs further (32) if you also push
# LEWM_NUM_TOKENS up. Without LGP the side-channel-only configuration
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
#            LGP-only ablations where the Latent Goal Predictor is the
#            sole le-wm pathway
LEWM_INJECT_TO   ?= suffix

# Latent Goal Predictor (LGP) — implementation of the "Future Sight" expert
# from design/future-sight-implicit-wm.md (Phase A). Adds a second
# flow-matching head next to the action expert that regresses to the
# encoded chunk-end frame z_{t+chunk_size} in le-wm's 192-dim latent. The
# active default for sawseenvlawm runs the *combined* configuration —
# le-wm side-channel into the action expert (LEWM_INJECT_TO=suffix) AND
# LGP enabled — to test whether the two pathways stack. Set LGP=false for
# a side-channel-only run, or LEWM_INJECT_TO=none LGP=true for an LGP-only
# ablation (which isolates the FS effect; see design/SawSeenVLAWM.md).
LGP              ?= true
LGP_LOSS_WEIGHT  ?= 1.0

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
	  -v $(CURDIR)/outputs:/lerobot/outputs \
	  -v $(CURDIR)/src:/lerobot/src \
	  -e MUJOCO_GL=egl \
	  -e HF_DATASETS_CACHE=/tmp/hf-datasets \
	  -e WANDB_API_KEY=$(WANDB_API_KEY) \
	  -e ACCELERATE_MIXED_PRECISION=$(MIXED_PRECISION) \
	  -e PYTHONPATH=/lewm-src:/lerobot/src \
	  $(DOCKER_CUDA_ENV) \
	  -w /lerobot \
	  $(DOCKER_IMAGE)

.PHONY: build train eval table

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
	  --policy.lgp_enabled=$(LGP) \
	  --policy.lgp_loss_weight=$(LGP_LOSS_WEIGHT) \
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
	  --tensorboard.enable=$(TENSORBOARD)

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

TABLE_RUN      ?= $(shell ls -td outputs/eval/*/* 2>/dev/null | head -1)
TABLE_LABEL    ?= Policy

table:
	@python3 eval_table.py $(TABLE_RUN) --label "$(TABLE_LABEL)" $(if $(TABLE_LATEX),--latex)
