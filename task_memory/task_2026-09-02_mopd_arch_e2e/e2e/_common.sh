#!/usr/bin/env bash
# Open-MOPD E2E — shared settings for every stage script.
#
# Source this, do not execute it.
set -euo pipefail

REPO="${REPO:-/data/ycfeng/Open-MOPD}"
VENV="${VENV:-/data/ycfeng/envs/openmopd-py312}"
WORK="${WORK:-/data/ycfeng/tmp/openmopd-e2e}"
ASSETS="${ASSETS:-$WORK/assets}"
RUNS="${RUNS:-$WORK/runs}"
LOGS="${LOGS:-$WORK/logs}"

export TMPDIR="${TMPDIR:-/data/ycfeng/tmp}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1          # hard guarantee: no weight download
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/training/verl:${PYTHONPATH:-}"

PY="$VENV/bin/python"
TORCHRUN="${TORCHRUN:-$VENV/bin/torchrun}"

MODEL_STUDENT="$ASSETS/models/student"
MODEL_T_MATH="$ASSETS/models/teacher_math"
MODEL_T_CODE="$ASSETS/models/teacher_code"
MODEL_T_IF="$ASSETS/models/teacher_if"

RL_TRAIN="$ASSETS/data/rl_train.parquet"
RL_VAL="$ASSETS/data/rl_val.parquet"
SFT_TRAIN="$ASSETS/data/sft_train.parquet"
SFT_VAL="$ASSETS/data/sft_val.parquet"
EVAL_INPUT="$ASSETS/data/eval_input.parquet"

OPD_VAL_REWARD="$REPO/training/verl/verl/utils/reward_score/opd_val_dispatch.py"

# Sequence budget. Kept tiny on purpose: the point is to exercise every code path,
# not to train anything. max_model_len must equal prompt+response.
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-128}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-64}"
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))

GPUS="${GPUS:-1}"
TRAIN_BS="${TRAIN_BS:-12}"
MINI_BS="${MINI_BS:-12}"
MICRO_BS="${MICRO_BS:-2}"
N_ROLLOUT="${N_ROLLOUT:-2}"
STEPS="${STEPS:-2}"

mkdir -p "$RUNS" "$LOGS"

# Shared PPO-family overrides: student model, data budget, batch shapes, 1-GPU vLLM.
#
# use_remove_padding=False on purpose — the padding-free path is the only one that
# needs flash-attn, which we deliberately did not install (see 00_setup_env.sh).
# tensor_model_parallel_size=1 overrides the rollout.yaml default of 2
# (verl/trainer/config/rollout/rollout.yaml:57), which would fail on 1 GPU.
common_ppo_overrides() {
    cat <<EOF
data.train_files=$RL_TRAIN
data.val_files=$RL_VAL
data.train_batch_size=$TRAIN_BS
data.max_prompt_length=$MAX_PROMPT_LEN
data.max_response_length=$MAX_RESPONSE_LEN
data.filter_overlong_prompts=True
data.truncation=error
data.dataloader_num_workers=0
actor_rollout_ref.model.path=$MODEL_STUDENT
actor_rollout_ref.model.use_remove_padding=False
+actor_rollout_ref.model.override_config.attn_implementation=eager
actor_rollout_ref.actor.strategy=fsdp
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BS
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BS
actor_rollout_ref.actor.optim.lr=1e-4
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.25
actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN
actor_rollout_ref.rollout.max_num_batched_tokens=1024
actor_rollout_ref.rollout.max_num_seqs=64
actor_rollout_ref.rollout.enforce_eager=True
actor_rollout_ref.rollout.n=$N_ROLLOUT
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BS
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BS
trainer.n_gpus_per_node=$GPUS
trainer.nnodes=1
trainer.total_epochs=1
trainer.total_training_steps=$STEPS
trainer.logger=['console']
trainer.val_before_train=False
trainer.test_freq=-1
trainer.resume_mode=disable
EOF
}

teacher_overrides() {
    # \$1 = teacher index (1-based, teacher 0 is reward_model.model.path)
    # \$2 = model path
    cat <<EOF
+mt_reward_model_$1.enable=True
+mt_reward_model_$1.model.path=$2
+mt_reward_model_$1.model.input_tokenizer=null
+mt_reward_model_$1.model.use_remove_padding=False
+mt_reward_model_$1.model.override_config.attn_implementation=eager
+mt_reward_model_$1.micro_batch_size_per_gpu=$MICRO_BS
EOF
}

banner() { printf '\n\033[1;36m===== %s =====\033[0m\n' "$*"; }

require_assets() {
    for p in "$MODEL_STUDENT" "$MODEL_T_MATH" "$MODEL_T_CODE" "$MODEL_T_IF" \
             "$RL_TRAIN" "$RL_VAL" "$SFT_TRAIN" "$EVAL_INPUT"; do
        [[ -e "$p" ]] || { echo "[common] missing asset: $p — run 10_make_dummy_assets.py first" >&2; exit 2; }
    done
}
