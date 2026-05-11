# Fine-tune SawSeenVLAKI on LIBERO and evaluate. All targets run inside the
# LIBERO benchmark Docker image so libero/MuJoCo/CUDA are pre-configured.
#
# SawSeenVLAKI = SawSeenVLA + Knowledge Insulation (KI) with FAST action
# tokens. The VLM is given a discrete next-action-token CE objective on
# top of the action expert's flow-matching MSE; the action expert reads
# detached VLM K/V so the flow-matching gradient never updates VLM
# weights — the VLM only learns "what actions look like" through CE,
# routed back via LoRA adapters. See design/SawSeenVLAKI.md and
# design/TODO.md item 2 for the architecture / motivation.
#
#   make -f sawseenvlaki.mk build   # build LIBERO benchmark Docker image (same image as sawseenvla.mk)
#   make -f sawseenvlaki.mk train   # fine-tune in the container
#   make -f sawseenvlaki.mk eval    # eval the trained policy in the container
#
# Override any variable on the command line, e.g.
#   make -f sawseenvlaki.mk train STEPS=80000 BATCH_SIZE=8
#   make -f sawseenvlaki.mk train KI=false   # off → behaves as SawSeenVLA
#   make -f sawseenvlaki.mk eval EVAL_POLICY=$(OUTPUT_DIR)/checkpoints/last/pretrained_model

DOCKER_IMAGE   ?= lerobot-benchmark-libero
HF_CACHE_DIR   ?= $(HOME)/.cache/huggingface
LIBERO_CACHE_DIR ?= $(HOME)/.cache/libero
GPU            ?=

# Train
DATASET_REPO   ?= HuggingFaceVLA/libero
OUTPUT_DIR     ?= outputs/train/sawseenvlaki_nodetach_kiw0.1_lora_r_16_libero_24k_bs32_2xGPUs_bf16
JOB_NAME       ?= sawseenvlaki_libero
STEPS          ?= 24000
# bs=32 (vs sawseenvla bs=64) compensates for the FAST tokens
# extending the prefix from ~113 to ~193 (with FAST_MAX_TOKENS=80) —
# attention is quadratic in seq, so the per-sample activation budget
# roughly doubles. Drop further if you bump FAST_MAX_TOKENS or
# chunk_size.
BATCH_SIZE     ?= 32
NUM_WORKERS    ?= 4
SAVE_FREQ      ?= 1000
LOG_FREQ       ?= 100
# Sqrt-scaled from the bs64 baseline (LR=4e-4 at global_batch=128):
# LR ≈ 4e-4 × sqrt(global_batch/128). Default tuned for BATCH_SIZE=32
# NUM_GPUS=2 (global_batch=64 → LR=2.83e-4).
LR             ?= 2.83e-4
COMPILE_MODEL   ?= false
COMPILE_MODE    ?= max-autotune
PAD_LANGUAGE_TO ?= max_length
DEVICE         ?= cuda
WANDB          ?= false
TENSORBOARD    ?= true
NUM_GPUS       ?= 2
MIXED_PRECISION ?= bf16

# PEFT / LoRA. KI requires LoRA — full-VLM training would OOM on 24 GB.
PEFT             ?= true
LORA_R           ?= 16

# Knowledge Insulation knobs.
# KI=true enables: (a) FAST CE head trained jointly with the action
# expert, (b) detach barrier on the VLM K/V tensors going into the
# action expert. Setting KI=false reduces the policy to a structurally
# identical SawSeenVLA (registered separately so checkpoints don't
# collide).
KI               ?= true
KI_LOSS_WEIGHT   ?= 0.1
# Whether to apply the KI detach barrier on the VLM K/V going into
# the action expert. true = paper-faithful KI (FM gradient never
# reaches VLM, VLM LoRA is adapted by CE alone). false = "auxiliary
# FAST head" mode: FM gradient still reaches VLM LoRA, CE is purely
# additive. Use false on LoRA-budget hardware where the small LoRA
# subspace can't be split between FM and CE without action loss
# stalling.
KI_DETACH        ?= false
# Right-pad bound for the FAST token sequence per chunk. Real lengths
# for LIBERO chunks (chunk_size=50, action_dim=7) span 100-145 tokens
# depending on chunk content; 160 covers the tail with margin and a
# truncation warning fires past that. Bumping this directly grows the
# VLM prefix and quadratically raises attention memory — re-tune
# BATCH_SIZE if you change this.
FAST_MAX_TOKENS  ?= 180
FAST_VOCAB_SIZE  ?= 2048

TRAIN_LAUNCHER  = $(if $(filter-out 1,$(NUM_GPUS)),accelerate launch --multi_gpu --num_processes=$(NUM_GPUS) --mixed_precision=$(MIXED_PRECISION) -m lerobot.scripts.lerobot_train,lerobot-train)
DOCKER_CUDA_ENV = $(if $(filter-out 1,$(NUM_GPUS)),-e CUDA_VISIBLE_DEVICES=$(shell python3 -c "print(','.join(str(i) for i in range($(NUM_GPUS))))"),)

# Eval
EVAL_POLICY    ?= $(OUTPUT_DIR)/checkpoints/last/pretrained_model
EVAL_TASKS     ?= libero_spatial,libero_object,libero_goal,libero_10
EVAL_EPISODES  ?= 10
EVAL_BATCH     ?= 10
EVAL_PARALLEL  ?= 1
EVAL_N_ACTION_STEPS ?= 10

DOCKER_RUN = docker run $(if $(GPU),--gpus device=$(GPU) -e MUJOCO_EGL_DEVICE_ID=0,--gpus all) --rm \
	  --shm-size=8g \
	  -v $(HF_CACHE_DIR):/home/user_lerobot/.cache/huggingface \
	  -v $(LIBERO_CACHE_DIR):/home/user_lerobot/.cache/libero \
	  -v $(CURDIR)/outputs:/lerobot/outputs \
	  -v $(CURDIR)/src:/lerobot/src \
	  -e MUJOCO_GL=egl \
	  -e HF_DATASETS_CACHE=/tmp/hf-datasets \
	  -e WANDB_API_KEY=$(WANDB_API_KEY) \
	  -e ACCELERATE_MIXED_PRECISION=$(MIXED_PRECISION) \
	  $(DOCKER_CUDA_ENV) \
	  -w /lerobot \
	  $(DOCKER_IMAGE)

.PHONY: build train eval

build:
	docker build -f docker/Dockerfile.benchmark.libero -t $(DOCKER_IMAGE) .

train:
	$(DOCKER_RUN) $(TRAIN_LAUNCHER) \
	  --policy.type=sawseenvlaki \
	  --policy.load_vlm_weights=true \
	  --policy.push_to_hub=false \
	  --policy.device=$(DEVICE) \
	  --policy.optimizer_lr=$(LR) \
	  --policy.scheduler_decay_steps=$(STEPS) \
	  --policy.compile_model=$(COMPILE_MODEL) \
	  --policy.compile_mode=$(COMPILE_MODE) \
	  --policy.pad_language_to=$(PAD_LANGUAGE_TO) \
	  --policy.ki_enabled=$(KI) \
	  --policy.ki_loss_weight=$(KI_LOSS_WEIGHT) \
	  --policy.ki_detach=$(KI_DETACH) \
	  --policy.fast_max_action_tokens=$(FAST_MAX_TOKENS) \
	  --policy.fast_vocab_size=$(FAST_VOCAB_SIZE) \
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
