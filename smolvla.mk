# Fine-tune SmolVLA on LIBERO and evaluate. All targets run inside the
# LIBERO benchmark Docker image so libero/MuJoCo/CUDA are pre-configured.
#
#   make -f smolvla.mk build   # build LIBERO benchmark Docker image (required first)
#   make -f smolvla.mk train   # fine-tune in the container
#   make -f smolvla.mk eval    # eval the trained policy in the container
#
# Override any variable on the command line, e.g.
#   make -f smolvla.mk train STEPS=80000 BATCH_SIZE=8
#   make -f smolvla.mk eval EVAL_POLICY=$(OUTPUT_DIR)/checkpoints/last/pretrained_model

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
# SmolVLM2 backbone weights from `lerobot/smolvla_base` (the recipe in
# docs/source/libero.mdx).
DATASET_REPO   ?= HuggingFaceVLA/libero
OUTPUT_DIR     ?= outputs/train/smolvla_libero_256k_bs_16_bf16
JOB_NAME       ?= smolvla_libero_ft
STEPS          ?= 256000
BATCH_SIZE     ?= 16
NUM_WORKERS    ?= 16
SAVE_FREQ      ?= 8000
LOG_FREQ       ?= 1000
# Peak LR (cosine schedule). Smolvla default is 1e-4; scale up by ~sqrt(N)
# when increasing the effective batch (e.g. 1.4e-4 for 2× batch).
LR             ?= 2.0e-4
DEVICE         ?= cuda
WANDB          ?= false
# NUM_GPUS=1 runs lerobot-train directly. NUM_GPUS>1 launches via
# `accelerate launch --multi_gpu`. BATCH_SIZE is the global batch — accelerate
# splits it across processes (each GPU sees BATCH_SIZE / NUM_GPUS samples).
# Requires GPU unset (so the container sees all GPUs).
NUM_GPUS       ?= 1
# Mixed-precision for training (Accelerator autocast). "no" = fp32 (default),
# "bf16" recommended on Ampere+ for ~1.5–1.7× forward/backward speedup and
# ~40–50% VRAM savings; "fp16" also accepted.
MIXED_PRECISION ?= no
TRAIN_LAUNCHER  = $(if $(filter-out 1,$(NUM_GPUS)),accelerate launch --multi_gpu --num_processes=$(NUM_GPUS) --mixed_precision=$(MIXED_PRECISION) -m lerobot.scripts.lerobot_train,lerobot-train)
# The base image bakes `CUDA_VISIBLE_DEVICES=0`; for multi-GPU we override it
# so all NUM_GPUS devices are visible to torch.cuda.
DOCKER_CUDA_ENV = $(if $(filter-out 1,$(NUM_GPUS)),-e CUDA_VISIBLE_DEVICES=$(shell python3 -c "print(','.join(str(i) for i in range($(NUM_GPUS))))"),)

# Eval
# Default: pi0.5 fine-tuned on HuggingFaceVLA/libero — drop-in compatible with
# `--env.type=libero`. Override with a local snapshot after training, e.g.
# `EVAL_POLICY=outputs/train/smolvla_libero_ft/checkpoints/last/pretrained_model`.
# EVAL_POLICY    ?= lerobot/pi05_libero_finetuned
# EVAL_POLICY    ?= lerobot/smolvla_base
EVAL_POLICY    ?= outputs/train/smolvla_libero_ft_128k_bf16/checkpoints/last/pretrained_model

EVAL_TASKS     ?= libero_spatial,libero_object,libero_goal,libero_10
EVAL_EPISODES  ?= 10
# Parallelism knobs. EVAL_BATCH = N parallel envs of the same task batched in
# policy inference. EVAL_PARALLEL = N tasks running concurrently (each spawns
# its own MuJoCo env + EGL context). Combined wall-time reduction is roughly
# EVAL_BATCH × EVAL_PARALLEL until CPU cores or GPU memory saturate.
EVAL_BATCH     ?= 10
EVAL_PARALLEL  ?= 1
# Number of actions consumed from each predicted chunk before re-querying the
# policy. Default 10 = full chunk (smolvla policy default). The SmolVLA paper
# (and pi05's libero.mdx recipe) sets this to 10 at inference time.
EVAL_N_ACTION_STEPS ?= 10

DOCKER_RUN = docker run $(if $(GPU),--gpus device=$(GPU) -e MUJOCO_EGL_DEVICE_ID=0,--gpus all) --rm \
	  --shm-size=8g \
	  -v $(HF_CACHE_DIR):/home/user_lerobot/.cache/huggingface \
	  -v $(LIBERO_CACHE_DIR):/home/user_lerobot/.cache/libero \
	  -v $(CURDIR)/outputs:/lerobot/outputs \
	  -e MUJOCO_GL=egl \
	  -e HF_DATASETS_CACHE=/tmp/hf-datasets \
	  -e WANDB_API_KEY=$(WANDB_API_KEY) \
	  -e ACCELERATE_MIXED_PRECISION=$(MIXED_PRECISION) \
	  $(DOCKER_CUDA_ENV) \
	  -w /lerobot \
	  $(DOCKER_IMAGE)

.PHONY: build train eval table

build:
	docker build -f docker/Dockerfile.benchmark.libero -t $(DOCKER_IMAGE) .

train:
	$(DOCKER_RUN) $(TRAIN_LAUNCHER) \
	  --policy.type=smolvla \
	  --policy.load_vlm_weights=true \
	  --policy.push_to_hub=false \
	  --policy.device=$(DEVICE) \
	  --policy.optimizer_lr=$(LR) \
	  --policy.scheduler_decay_steps=$(STEPS) \
	  --dataset.repo_id=$(DATASET_REPO) \
	  --output_dir=$(OUTPUT_DIR) \
	  --job_name=$(JOB_NAME) \
	  --steps=$(STEPS) \
	  --batch_size=$(BATCH_SIZE) \
	  --num_workers=$(NUM_WORKERS) \
	  --save_freq=$(SAVE_FREQ) \
	  --log_freq=$(LOG_FREQ) \
	  --eval_freq=$(STEPS) \
	  --wandb.enable=$(WANDB)

eval:
	$(DOCKER_RUN) lerobot-eval \
	  --policy.path=$(EVAL_POLICY) \
	  --policy.device=$(DEVICE) \
	  --policy.n_action_steps=$(EVAL_N_ACTION_STEPS) \
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
