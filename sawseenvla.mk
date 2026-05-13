# Fine-tune SawSeenVLA on LIBERO and evaluate. All targets run inside the
# LIBERO benchmark Docker image so libero/MuJoCo/CUDA are pre-configured.
#
# SawSeenVLA is a structural clone of SmolVLA (same SmolVLM2-500M backbone, same
# flow-matching action expert, identical defaults) registered as a separate
# policy under `--policy.type=sawseenvla`.
#
#   make -f sawseenvla.mk build   # build LIBERO benchmark Docker image (required first; same image as smolvla.mk)
#   make -f sawseenvla.mk train   # fine-tune in the container
#   make -f sawseenvla.mk eval    # eval the trained policy in the container
#
# Override any variable on the command line, e.g.
#   make -f sawseenvla.mk train STEPS=80000 BATCH_SIZE=8
#   make -f sawseenvla.mk eval EVAL_POLICY=$(OUTPUT_DIR)/checkpoints/last/pretrained_model

DOCKER_IMAGE   ?= lerobot-benchmark-libero
HF_CACHE_DIR   ?= $(HOME)/.cache/huggingface
# Persistent libero asset cache (~408 MB). The libero pip package's
# get_assets_path() only checks <pkg>/assets and otherwise re-downloads to
# ~/.cache/libero/assets — bind-mounting that path keeps the assets on the host.
LIBERO_CACHE_DIR ?= $(HOME)/.cache/libero
# Optional: pin to a specific host GPU index (e.g. GPU=1 to run eval on GPU 1
# while training is on GPU 0). When set, the container sees only that GPU as
# device 0, so MUJOCO_EGL_DEVICE_ID is forced to 0.
GPU            ?=

# Train
# Fresh action expert against the dataset's features, loading only the
# SmolVLM2 backbone weights (the recipe in docs/source/libero.mdx, applied to
# the sawseenvla policy registration).
DATASET_REPO   ?= HuggingFaceVLA/libero
OUTPUT_DIR     ?= outputs/train/sawseenvla_lora_r_16_libero_8k_bs96_2xGPUs_bf16
JOB_NAME       ?= sawseenvla_libero
STEPS          ?= 8000
BATCH_SIZE     ?= 96
NUM_WORKERS    ?= 4
SAVE_FREQ      ?= 1000
LOG_FREQ       ?= 100
# Peak LR (cosine schedule). Square-root scaled with global batch from the
# bs64 baseline (LR=4e-4 at global_batch=128): LR ≈ 4e-4 × sqrt(global_batch/128).
# Default is tuned for BATCH_SIZE=96, NUM_GPUS=2 (global_batch=192 → LR≈5e-4).
LR             ?= 5.0e-4
# Compile knobs. compile_model=true with max-autotune buys ~1.35× throughput,
# BUT requires pad_language_to=max_length — with "longest", per-batch language
# token length changes shape → dynamo recompile storm → 5-15× slowdown.
# Set COMPILE_MODEL=false for fast smoke runs (skips ~10-min compile warmup).
COMPILE_MODEL   ?= false
COMPILE_MODE    ?= max-autotune
PAD_LANGUAGE_TO ?= max_length
DEVICE         ?= cuda
WANDB          ?= false
TENSORBOARD    ?= true
# PEFT (LoRA) — when true, LoRA adapters land on the frozen VLM's
# attention Q/V projections (~2M trainable params at r=16); the action
# expert and small projections train fully via modules_to_save.
# See _get_default_peft_targets in modeling_sawseenvla.py.
PEFT             ?= true
LORA_R           ?= 16
# NUM_GPUS=1 runs lerobot-train directly. NUM_GPUS>1 launches via
# `accelerate launch --multi_gpu`. BATCH_SIZE is the global batch — accelerate
# splits it across processes (each GPU sees BATCH_SIZE / NUM_GPUS samples).
# Requires GPU unset (so the container sees all GPUs).
NUM_GPUS       ?= 2
# Mixed-precision for training (Accelerator autocast). "no" = fp32 (default),
# "bf16" recommended on Ampere+ for ~1.5–1.7× forward/backward speedup and
# ~40–50% VRAM savings; "fp16" also accepted.
MIXED_PRECISION ?= bf16
TRAIN_LAUNCHER  = $(if $(filter-out 1,$(NUM_GPUS)),accelerate launch --multi_gpu --num_processes=$(NUM_GPUS) --mixed_precision=$(MIXED_PRECISION) -m lerobot.scripts.lerobot_train,lerobot-train)
# The base image bakes `CUDA_VISIBLE_DEVICES=0`; for multi-GPU we override it
# so all NUM_GPUS devices are visible to torch.cuda.
DOCKER_CUDA_ENV = $(if $(filter-out 1,$(NUM_GPUS)),-e CUDA_VISIBLE_DEVICES=$(shell python3 -c "print(','.join(str(i) for i in range($(NUM_GPUS))))"),)

# Eval
# Default points at a sawseenvla checkpoint produced by `make -f sawseenvla.mk train`.
# Override with a different snapshot via:
# `EVAL_POLICY=outputs/train/<your_run>/checkpoints/last/pretrained_model`.
# NOTE: hub checkpoints saved as `type: smolvla` (e.g. lerobot/smolvla_base) will
# NOT load via this Makefile because `from_pretrained` dispatches on the choice
# key. Train sawseenvla with `--policy.load_vlm_weights=true` to bootstrap from
# the SmolVLM2 backbone instead.
# EVAL_POLICY    ?= outputs/train/sawseenvla_libero_96k_bs64_2xGPUs_bf16/checkpoints/last/pretrained_model
# EVAL_POLICY    ?= outputs/train/sawseenvla_libero_96k_bs96_compile_2xGPUs_bf16/checkpoints/last/pretrained_model
EVAL_POLICY    ?= outputs/train/sawseenvla_libero_8k_bs96_2xGPUs_bf16/checkpoints/last/pretrained_model

EVAL_TASKS     ?= libero_spatial,libero_object,libero_goal,libero_10
EVAL_EPISODES  ?= 10
# Parallelism knobs. EVAL_BATCH = N parallel envs of the same task batched in
# policy inference. EVAL_PARALLEL = N tasks running concurrently (each spawns
# its own MuJoCo env + EGL context). Combined wall-time reduction is roughly
# EVAL_BATCH × EVAL_PARALLEL until CPU cores or GPU memory saturate.
EVAL_BATCH     ?= 10
EVAL_PARALLEL  ?= 1
# Number of actions consumed from each predicted chunk before re-querying the
# policy. Default 10 = full chunk (sawseenvla policy default, matches smolvla).
# The SmolVLA paper (and pi05's libero.mdx recipe) sets this to 10 at
# inference time.
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

.PHONY: build train eval table mine

build:
	docker build -f docker/Dockerfile.benchmark.libero -t $(DOCKER_IMAGE) .

train:
	$(DOCKER_RUN) $(TRAIN_LAUNCHER) \
	  --policy.type=sawseenvla \
	  --policy.load_vlm_weights=true \
	  --policy.push_to_hub=false \
	  --policy.device=$(DEVICE) \
	  --policy.optimizer_lr=$(LR) \
	  --policy.scheduler_decay_steps=$(STEPS) \
	  --policy.compile_model=$(COMPILE_MODEL) \
	  --policy.compile_mode=$(COMPILE_MODE) \
	  --policy.pad_language_to=$(PAD_LANGUAGE_TO) \
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

# Force compile_model=false at eval — the saved policy config has
# compile_model=true from training, and from_pretrained would otherwise trigger
# a ~10-min compile warmup that doesn't amortize over libero rollouts (and
# would re-compile on the eval-time batch shape anyway).
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

# Print a docs-format result row from an eval_info.json. Defaults to the most
# recent eval run; override with TABLE_RUN=<path>. Set TABLE_LATEX=1 for LaTeX.
TABLE_RUN      ?= $(shell ls -td outputs/eval/*/* 2>/dev/null | head -1)
TABLE_LABEL    ?= Policy

table:
	@python3 eval_table.py $(TABLE_RUN) --label "$(TABLE_LABEL)" $(if $(TABLE_LATEX),--latex)

# ─── Mining: roll out SawSeenVLA checkpoints in LIBERO sim, record trajectories
# in LeRobot v3 format for downstream LeWM training. Mixes three checkpoints
# (early/mid/late) to seed off-expert state coverage. Output schema matches
# HuggingFaceVLA/libero so it can be concatenated with the expert dataset by
# le-wm/scripts/libero_to_h5.py (multi-source support).
#
# Host paths for the SawSeenVLA training run and the output dataset (defaults
# match the user's local setup; override on the command line). MINE_CKPT_ROOT
# must contain the `<step>/pretrained_model/` subdirs referenced by MINE_CKPTS.
MINE_CKPT_ROOT   ?= /mnt/hard_disk_16Tb/data/lerobot/outputs/train/sawseenvla_libero_8k_bs96_2xGPUs_bf16/checkpoints
MINE_CKPTS       ?= 002000 004000 006000
MINE_EPS_PER_TASK ?= 4 3 3
MINE_SUITES      ?= libero_spatial libero_object libero_goal libero_10
MINE_OUTPUT_HOST ?= /mnt/hard_disk_16Tb/data/lerobot/datasets/sawseenvla_libero_mined
MINE_REPO_ID     ?= local/sawseenvla_libero_mined
MINE_SEED        ?= 0
# Optional: restrict to a few task_ids per suite for smoke-testing
# (e.g. MINE_TASK_IDS="0 1" mines only the first two tasks of each suite).
MINE_TASK_IDS    ?=

# iCEM-style colored noise injected on top of the VLA's action per step.
# Pure-policy rollouts when std=0. Recommended starting point: std=0.1, beta=2.
# Use --use-async-envs flag for AsyncVectorEnv when n_envs per task is large
# (subprocess per env; cleaner EGL contexts, higher throughput at the cost of
# extra startup time per task).
MINE_NOISE_STD   ?= 0.0
MINE_NOISE_BETA  ?= 2.0
MINE_USE_ASYNC   ?= false

# We bind-mount the PARENT directory and let the container create the leaf
# itself — LeRobotDataset.create()'s mkdir(exist_ok=False) chokes on a
# pre-existing leaf, and the container user (UID 1001) can't rmdir host-owned
# (UID 1000) bind-mount artifacts. Split the host path into parent/leaf.
MINE_OUTPUT_PARENT = $(patsubst %/,%,$(dir $(MINE_OUTPUT_HOST)))
MINE_OUTPUT_NAME   = $(notdir $(MINE_OUTPUT_HOST))

# Build the container-side ckpt paths from MINE_CKPTS (each step → /ckpts/<step>/pretrained_model).
MINE_CKPT_PATHS  = $(foreach s,$(MINE_CKPTS),/ckpts/$(s)/pretrained_model)

mine:
	@mkdir -p $(MINE_OUTPUT_PARENT)
	@# Make the parent world-writable so the container user (user_lerobot, UID 1001)
	@# can mkdir the leaf alongside the host's UID 1000 ownership. Swallow
	@# failures for already-writable system dirs (e.g. /tmp, sticky bit 1777).
	@chmod 777 $(MINE_OUTPUT_PARENT) 2>/dev/null || true
	docker run $(if $(GPU),--gpus device=$(GPU) -e MUJOCO_EGL_DEVICE_ID=0,--gpus all) --rm \
	  --shm-size=8g \
	  -v $(HF_CACHE_DIR):/home/user_lerobot/.cache/huggingface \
	  -v $(LIBERO_CACHE_DIR):/home/user_lerobot/.cache/libero \
	  -v $(MINE_CKPT_ROOT):/ckpts:ro \
	  -v $(MINE_OUTPUT_PARENT):/datasets \
	  -v $(CURDIR)/src:/lerobot/src \
	  -v $(CURDIR)/scripts:/lerobot/scripts \
	  -e MUJOCO_GL=egl \
	  -e HF_DATASETS_CACHE=/tmp/hf-datasets \
	  $(DOCKER_CUDA_ENV) \
	  -w /lerobot \
	  $(DOCKER_IMAGE) \
	  python scripts/mine_libero.py \
	    --ckpts $(MINE_CKPT_PATHS) \
	    --eps-per-task $(MINE_EPS_PER_TASK) \
	    --suites $(MINE_SUITES) \
	    --output-root /datasets/$(MINE_OUTPUT_NAME) \
	    --repo-id $(MINE_REPO_ID) \
	    --seed $(MINE_SEED) \
	    --action-noise-std $(MINE_NOISE_STD) \
	    --noise-beta $(MINE_NOISE_BETA) \
	    $(if $(filter true,$(MINE_USE_ASYNC)),--use-async-envs,) \
	    $(if $(MINE_TASK_IDS),--task-ids $(MINE_TASK_IDS))
