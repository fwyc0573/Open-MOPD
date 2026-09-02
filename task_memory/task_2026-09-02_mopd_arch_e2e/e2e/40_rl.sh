#!/usr/bin/env bash
# Open-MOPD E2E — Step 40: domain-RL (GRPO) stage, via the released launcher.
#
# Entrypoint: scripts/local/rl.sh --run  ->  python -m verl.trainer.main_ppo
# The real recipe runs this three times (math / code / IF) to produce the three
# teachers. Here one short run proves the verifiable-reward path.
#
# rl.sh already emits: algorithm.adv_estimator=grpo, rollout.name=vllm,
# reward_model.enable=False, rollout.max_model_len=P+R, trainer.logger=['console'].
# Everything else has to arrive through --extra, because the shipped defaults are
# tuned for an 8-GPU 7B run and hard-fail on a single-GPU tiny model:
#   * rollout.tensor_model_parallel_size defaults to 2 (rollout.yaml:57) -> needs 1
#   * actor.ppo_mini_batch_size defaults to 256 (actor.yaml:15) > train_batch_size
#   * actor.ppo_micro_batch_size{,_per_gpu} are both null -> ActorConfig asserts
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
require_assets

OUT="$RUNS/rl"
LOG="$LOGS/40_rl.log"
mkdir -p "$OUT/checkpoints"

banner "40 — domain RL (GRPO) via scripts/local/rl.sh --run"
set +e
env PYTHON_BIN="$PY" \
    TRAIN_BATCH_SIZE="$TRAIN_BS" \
    MAX_PROMPT_LENGTH="$MAX_PROMPT_LEN" MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LEN" \
    N_RESPONSES="$N_ROLLOUT" TOTAL_EPOCHS=1 ADV_ESTIMATOR=grpo \
    REWARD_FUNCTION_PATH="$OPD_VAL_REWARD" REWARD_FUNCTION_NAME=reward_func \
    PROJECT_NAME=OpenMOPD-e2e EXPERIMENT_NAME=rl-dummy \
    bash "$REPO/scripts/local/rl.sh" --run \
        --model "$MODEL_STUDENT" \
        --train "$RL_TRAIN" --val "$RL_VAL" \
        --output "$OUT" --checkpoint "$OUT/checkpoints" --gpus "$GPUS" \
        --extra "actor_rollout_ref.model.use_remove_padding=False" \
        --extra "+actor_rollout_ref.model.override_config.attn_implementation=eager" \
        --extra "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BS" \
        --extra "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BS" \
        --extra "actor_rollout_ref.actor.optim.lr=1e-4" \
        --extra "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
        --extra "actor_rollout_ref.rollout.gpu_memory_utilization=0.30" \
        --extra "actor_rollout_ref.rollout.max_num_batched_tokens=1024" \
        --extra "actor_rollout_ref.rollout.max_num_seqs=64" \
        --extra "actor_rollout_ref.rollout.enforce_eager=True" \
        --extra "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BS" \
        --extra "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BS" \
        --extra "trainer.total_training_steps=$STEPS" \
        --extra "trainer.val_before_train=False" \
        --extra "trainer.test_freq=-1" \
        --extra "trainer.save_freq=$STEPS" \
        --extra "trainer.resume_mode=disable" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

banner "40 — evidence"
echo "[40] exit=$rc"
echo "[40] per-step metric lines:"
grep -E "^step:|step:[0-9]+ " "$LOG" | tail -4 || true
echo "[40] reward / loss:"
grep -oE "(critic/(score|rewards)/mean|actor/pg_loss|actor/grad_norm|perf/[a-z_]+):[-0-9.e+]+" "$LOG" | tail -20 || true
echo "[40] checkpoint tree:"; find "$OUT/checkpoints" -maxdepth 3 2>/dev/null | head -30 || true
exit $rc
