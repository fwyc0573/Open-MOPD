#!/usr/bin/env bash
# Open-MOPD E2E — Step 50: single-teacher on-policy distillation (OPD).
#
# Two parts, both recorded:
#   50a  the repaired local launcher, scripts/local/opd.sh --run.
#   50b  the equivalent corrected direct invocation, so the algorithm path remains
#        independently reproducible.
#
# What makes it distillation rather than RL:
#   algorithm.adv_estimator=token_reward_direct
#       core_algos.py:876 registers it; :899 sets advantages = token_level_rewards *
#       response_mask, with response_mask.unsqueeze(-1) applied only when the reward
#       tensor is 3-D (:897-898) — i.e. exactly the dense top-K case.
#   +actor_rollout_ref.rollout.reward_mode=opd_kl
#       ray_trainer.py:86-92 reads it, then deletes it so the RolloutConfig dataclass
#       conversion (fsdp_workers.py) does not choke on an unknown field.
#   +actor_rollout_ref.rollout.log_prob_top_k=K
#       ray_trainer.py:1968 reads it with default 0; at 0 the dense teacher path is
#       skipped entirely (the branch at :2232 tests top_k > 0).
#   reward_model.model.input_tokenizer=null
#       reward_model.yaml:33 defaults it to the student path, i.e. NON-null, which sets
#       _do_switch_chat_template=True (fsdp_workers.py:1810-1814) and routes the teacher
#       through code that indexes non_tensor_batch["raw_prompt"] (:2513-2519) — absent
#       unless data.return_raw_chat=True (legacy_data.yaml:59 defaults False).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
require_assets

TOP_K="${TOP_K:-16}"
OUT="$RUNS/opd"
LOG_A="$LOGS/50a_opd_shipped_launcher.log"
LOG_B="$LOGS/50b_opd_corrected.log"
mkdir -p "$OUT/checkpoints"

banner "50a — REPAIRED scripts/local/opd.sh --run"
set +e
env PYTHON_BIN="$PY" TRAIN_BATCH_SIZE="$TRAIN_BS" \
    MAX_PROMPT_LENGTH="$MAX_PROMPT_LEN" MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LEN" \
    N_RESPONSES="$N_ROLLOUT" TOTAL_EPOCHS=1 REWARD_MODEL_PATH="$MODEL_T_MATH" \
    bash "$REPO/scripts/local/opd.sh" --run \
        --model "$MODEL_STUDENT" --teacher "$MODEL_T_MATH" \
        --train "$RL_TRAIN" --val "$RL_VAL" \
        --output "$RUNS/opd-shipped" --gpus "$GPUS" 2>&1 | tee "$LOG_A"
rc_a=${PIPESTATUS[0]}
set -e
echo "[50a] repaired launcher exit=$rc_a"

banner "50b — CORRECTED direct invocation (reward_mode=opd_kl, top_k=$TOP_K)"
cd "$REPO/training"
mapfile -t OV < <(common_ppo_overrides)
set +e
"$PY" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=token_reward_direct \
    "${OV[@]}" \
    "+actor_rollout_ref.rollout.reward_mode=opd_kl" \
    "+actor_rollout_ref.rollout.log_prob_top_k=$TOP_K" \
    reward_model.enable=True \
    reward_model.strategy=fsdp \
    reward_model.model.path="$MODEL_T_MATH" \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=False \
    +reward_model.model.override_config.attn_implementation=eager \
    reward_model.micro_batch_size_per_gpu="$MICRO_BS" \
    custom_reward_function.path="$OPD_VAL_REWARD" \
    custom_reward_function.name=reward_func \
    trainer.default_local_dir="$OUT/checkpoints" \
    trainer.project_name=OpenMOPD-e2e \
    trainer.experiment_name=opd-dummy 2>&1 | tee "$LOG_B"
rc_b=${PIPESTATUS[0]}
set -e

banner "50 — evidence"
echo "[50a] shipped launcher exit=$rc_a"
echo "[50b] corrected run   exit=$rc_b"
echo
echo "--- distillation reward / actor metrics ---"
grep -oE "(critic/(score|rewards)/(mean|max|min)|actor/(pg_loss|grad_norm|pg_clipfrac|entropy)|teacher[a-z_/]*|opd[a-z_/]*):[-0-9.e+]+" "$LOG_B" | tail -25 || true
echo
echo "--- per-step lines ---"; grep -E "step:[0-9]+" "$LOG_B" | tail -3 || true
echo "[50] checkpoint tree:"; find "$OUT/checkpoints" -maxdepth 3 2>/dev/null | head -30 || true
exit $rc_b
