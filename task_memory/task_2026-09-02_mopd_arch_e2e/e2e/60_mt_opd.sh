#!/usr/bin/env bash
# Open-MOPD E2E — Step 60: MULTI-teacher on-policy distillation. The core of the paper.
#
# Two parts, both recorded:
#   60a  the repaired local launcher, scripts/local/mt_opd.sh --run.
#   60b  the corrected direct invocation, three teachers, three domains.
#
# The MT-OPD switches:
#   +actor_rollout_ref.rollout.reward_mode=mt_opd   ray_trainer.py:644 -> use_mt_opd
#   +actor_rollout_ref.rollout.log_prob_top_k=K     ray_trainer.py:1973-1974 RAISES at 0
#   +mt_opd.teacher_domains=[math,code,if]          ray_trainer.py:662, raise at :666
#   +mt_opd.n_additional_teachers=2                 ray_trainer.py:663, used at :1428
#   +mt_reward_model_{1,2}.*                        ray_trainer.py:1427-1440 demands them
#
# Teacher ordering is POSITIONAL and only the COUNT is checked (ray_trainer.py:2121-2126):
#   teacher 0 = reward_model.model.path            <- teacher_domains[0]
#   teacher i = mt_reward_model_{i}.model.path     <- teacher_domains[i]
# Swapping two teacher paths silently routes math prompts to the code teacher.
#
# Dataset requirement: a top-level `domain` column. ray_trainer.py:2110-2114 raises
# without it — AFTER every teacher forward has already run. An unknown domain label does
# NOT raise: mt_opd.py:57-61 hands it a uniform 1/N teacher row.
#
# MECHANISM SELECTION. The shipped mt_opd.sh sets none of M1/M2/M3, so as released it is
# naive M-OPD (routing only) and compute_domain_loss_weights returns None
# (mt_opd.py:207-208). This script runs both configurations so the difference is visible:
#   MT_MODE=naive   routing only
#   MT_MODE=m1      + target gradient shares            (the paper's budget control)
#   MT_MODE=full    + M2 reward-scale + M3 conflict mask
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
require_assets

TOP_K="${TOP_K:-16}"
MT_MODE="${MT_MODE:-m1}"
OUT="$RUNS/mt_opd-$MT_MODE"
LOG_A="$LOGS/60a_mt_opd_shipped_launcher.log"
LOG_B="$LOGS/60b_mt_opd_corrected_$MT_MODE.log"
mkdir -p "$OUT/checkpoints"

banner "60a — REPAIRED scripts/local/mt_opd.sh --run"
set +e
env PYTHON_BIN="$PY" TRAIN_BATCH_SIZE="$TRAIN_BS" \
    MAX_PROMPT_LENGTH="$MAX_PROMPT_LEN" MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LEN" \
    N_RESPONSES="$N_ROLLOUT" TOTAL_EPOCHS=1 \
    bash "$REPO/scripts/local/mt_opd.sh" --run \
        --model "$MODEL_STUDENT" \
        --teacher "$MODEL_T_MATH" --teacher "$MODEL_T_CODE" --teacher "$MODEL_T_IF" \
        --domains math,code,if \
        --train "$RL_TRAIN" --val "$RL_VAL" \
        --output "$RUNS/mt_opd-shipped" --gpus "$GPUS" 2>&1 | tee "$LOG_A"
rc_a=${PIPESTATUS[0]}
set -e
echo "[60a] repaired launcher exit=$rc_a"

# ---- assemble the mechanism overrides
MECH=()
case "$MT_MODE" in
    naive) ;;
    m1)
        MECH+=("+mt_opd.target_share_domains=[math,code,if]"
               "+mt_opd.target_share_values=[0.4,0.4,0.2]")
        ;;
    full)
        MECH+=("+mt_opd.target_share_domains=[math,code,if]"
               "+mt_opd.target_share_values=[0.4,0.4,0.2]"
               "+mt_opd.normalize_reward_scale=1.0"
               "+mt_opd.reward_scale_stat=mean"
               # multiply, not the `divide` default: mt_opd.py:290-296 records divide as
               # measured-harmful, and only multiply clamps (to [0.05, 20]).
               "+mt_opd.reward_scale_direction=multiply"
               "+mt_opd.conflict_policy=mask"
               "+mt_opd.conflict_nats=1.0")
        ;;
    *) echo "[60] unknown MT_MODE=$MT_MODE (naive|m1|full)" >&2; exit 2 ;;
esac

banner "60b — CORRECTED MT-OPD: MT_MODE=$MT_MODE, top_k=$TOP_K, 3 teachers / 3 domains"
cd "$REPO/training"
mapfile -t OV < <(common_ppo_overrides)
mapfile -t T1 < <(teacher_overrides 1 "$MODEL_T_CODE")
mapfile -t T2 < <(teacher_overrides 2 "$MODEL_T_IF")

set +e
"$PY" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=token_reward_direct \
    "${OV[@]}" \
    "+actor_rollout_ref.rollout.reward_mode=mt_opd" \
    "+actor_rollout_ref.rollout.log_prob_top_k=$TOP_K" \
    reward_model.enable=True \
    reward_model.strategy=fsdp \
    reward_model.model.path="$MODEL_T_MATH" \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=False \
    +reward_model.model.override_config.attn_implementation=eager \
    reward_model.micro_batch_size_per_gpu="$MICRO_BS" \
    "+mt_opd.teacher_domains=[math,code,if]" \
    "+mt_opd.n_additional_teachers=2" \
    "+mt_opd.domain_weighting=domain_routing" \
    "${MECH[@]}" "${T1[@]}" "${T2[@]}" \
    custom_reward_function.path="$OPD_VAL_REWARD" \
    custom_reward_function.name=reward_func \
    trainer.default_local_dir="$OUT/checkpoints" \
    trainer.project_name=OpenMOPD-e2e \
    trainer.experiment_name="mt-opd-dummy-$MT_MODE" 2>&1 | tee "$LOG_B"
rc_b=${PIPESTATUS[0]}
set -e

banner "60 — evidence: per-domain optimization budget (the paper's central measurement)"
echo "[60a] shipped launcher exit=$rc_a"
echo "[60b] corrected run    exit=$rc_b   MT_MODE=$MT_MODE"
echo
echo "--- mt_opd/domain/*  (prompt_share vs token_share is the diagnosis; loss_weight is the fix) ---"
grep -oE "mt_opd/domain/[A-Za-z_]+/[a-z_0-9]+:[-0-9.e+]+" "$LOG_B" | sort -u || echo "  (none)"
echo
echo "--- mt_opd/conflict/*  (free cross-teacher disagreement signal) ---"
grep -oE "mt_opd/conflict/[a-z_]+:[-0-9.e+]+" "$LOG_B" | sort -u || echo "  (none)"
echo
echo "--- mt_opd top-level ---"
grep -oE "mt_opd/[a-z_]+:[-0-9.e+]+" "$LOG_B" | sort -u || echo "  (none)"
echo
echo "--- actor / critic ---"
grep -oE "(actor/(pg_loss|grad_norm|pg_clipfrac|entropy)|critic/(score|rewards)/mean):[-0-9.e+]+" "$LOG_B" | tail -12 || true
echo
echo "[60] checkpoint tree:"; find "$OUT/checkpoints" -maxdepth 3 2>/dev/null | head -30 || true
exit $rc_b
