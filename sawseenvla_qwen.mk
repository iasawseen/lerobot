# Train SawSeenVLA-Qwen (Qwen3.5-0.8B + independent action expert, Scheme B
# per-layer cross-attention to anchor hidden states [4, 8, 12, 16, 20, 24]).
#
#   make -f sawseenvla_qwen.mk train   # train the expert (Qwen frozen by default)
#   make -f sawseenvla_qwen.mk eval    # eval on LIBERO suites

DOCKER_IMAGE      ?= lerobot-benchmark-libero
HF_CACHE_DIR      ?= /home/lucius/data/hf_cache_qwen
LIBERO_CACHE_DIR  ?= $(HOME)/.cache/libero
# Pin to a host GPU index (the container sees only that GPU as device 0).
GPU               ?=

# Train
DATASET_REPO     ?= HuggingFaceVLA/libero
DATASET_ROOT     ?=
DATASET_EPISODES ?=
OUTPUT_DIR       ?= outputs/train/sawseenvla_qwen_libero_8k_bs32_1xGPU_full_bf16
JOB_NAME         ?= sawseenvla_qwen_libero
STEPS            ?= 8000
BATCH_SIZE       ?= 32
NUM_WORKERS      ?= 4
SAVE_FREQ        ?= 1000
LOG_FREQ         ?= 100
LR               ?= 1.0e-4
COMPILE_MODEL    ?= false
COMPILE_MODE     ?= max-autotune
PAD_LANGUAGE_TO  ?= longest
DEVICE           ?= cuda
WANDB            ?= false
TENSORBOARD      ?= true
MIXED_PRECISION  ?= bf16

# Eval
EVAL_POLICY      ?= $(OUTPUT_DIR)/checkpoints/last/pretrained_model
EVAL_TASKS       ?= libero_spatial,libero_object,libero_goal,libero_10
EVAL_EPISODES    ?= 10
EVAL_BATCH       ?= 10
EVAL_PARALLEL    ?= 1
EVAL_N_ACTION_STEPS ?= 10

DOCKER_RUN = docker run $(if $(GPU),--gpus device=$(GPU) -e MUJOCO_EGL_DEVICE_ID=0,--gpus all) --rm \
	  --shm-size=8g \
	  -v $(HF_CACHE_DIR):/hf_cache \
	  -v $(LIBERO_CACHE_DIR):/home/user_lerobot/.cache/libero \
	  -v $(CURDIR)/outputs:/lerobot/outputs \
	  -v $(CURDIR)/src:/lerobot/src \
	  $(if $(DATASET_ROOT),-v $(DATASET_ROOT):$(DATASET_ROOT):ro,) \
	  -e MUJOCO_GL=egl \
	  -e HOME=/hf_cache \
	  -e HF_HOME=/hf_cache \
	  -e HF_DATASETS_CACHE=/tmp/hf-datasets \
	  -e WANDB_API_KEY=$(WANDB_API_KEY) \
	  -e ACCELERATE_MIXED_PRECISION=$(MIXED_PRECISION) \
	  -w /lerobot \
	  $(DOCKER_IMAGE)

.PHONY: train eval

train:
	$(DOCKER_RUN) lerobot-train \
	  --policy.type=sawseenvla_qwen \
	  --policy.load_vlm_weights=true \
	  --policy.push_to_hub=false \
	  --policy.device=$(DEVICE) \
	  --policy.optimizer_lr=$(LR) \
	  --policy.scheduler_decay_steps=$(STEPS) \
	  --policy.compile_model=$(COMPILE_MODEL) \
	  --policy.compile_mode=$(COMPILE_MODE) \
	  --policy.pad_language_to=$(PAD_LANGUAGE_TO) \
	  --dataset.repo_id=$(DATASET_REPO) \
	  $(if $(DATASET_ROOT),--dataset.root=$(DATASET_ROOT),) \
	  $(if $(DATASET_EPISODES),--dataset.episodes='$(DATASET_EPISODES)',) \
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
