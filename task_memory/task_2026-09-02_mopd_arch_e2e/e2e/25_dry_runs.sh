#!/usr/bin/env bash
# Open-MOPD E2E — Step 25: exercise all five local launchers in dry-run mode.
#
# The launchers are dry-run by default (scripts/local/common.sh:local_run_command).
# This step proves the argv each one builds, which is the contract between the
# shell layer and the Hydra layer. Nothing is executed and no GPU is touched.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
require_assets

LOG="$LOGS/25_dry_runs.log"
: > "$LOG"

run_dry() {
    local label="$1"; shift
    banner "25 — dry-run: $label"
    { echo "### $label"; echo "\$ $*"; } | tee -a "$LOG"
    ( PYTHON_BIN="$PY" TORCHRUN_BIN="$TORCHRUN" "$@" ) 2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
}

run_dry "sft.sh --help"     bash "$REPO/scripts/local/sft.sh" --help
run_dry "rl.sh --help"      bash "$REPO/scripts/local/rl.sh" --help
run_dry "opd.sh --help"     bash "$REPO/scripts/local/opd.sh" --help
run_dry "mt_opd.sh --help"  bash "$REPO/scripts/local/mt_opd.sh" --help
run_dry "eval.sh --help"    bash "$REPO/scripts/local/eval.sh" --help

run_dry "sft.sh" bash "$REPO/scripts/local/sft.sh" \
    --model "$MODEL_STUDENT" --train "$SFT_TRAIN" --val "$SFT_VAL" \
    --output "$RUNS/dry-sft" --gpus "$GPUS"

run_dry "rl.sh" bash "$REPO/scripts/local/rl.sh" \
    --model "$MODEL_STUDENT" --train "$RL_TRAIN" --val "$RL_VAL" \
    --output "$RUNS/dry-rl" --gpus "$GPUS"

run_dry "opd.sh" env REWARD_MODEL_PATH="$MODEL_T_MATH" \
    bash "$REPO/scripts/local/opd.sh" \
    --model "$MODEL_STUDENT" --teacher "$MODEL_T_MATH" \
    --train "$RL_TRAIN" --val "$RL_VAL" --output "$RUNS/dry-opd" --gpus "$GPUS"

run_dry "mt_opd.sh" bash "$REPO/scripts/local/mt_opd.sh" \
    --model "$MODEL_STUDENT" \
    --teacher "$MODEL_T_MATH" --teacher "$MODEL_T_CODE" --teacher "$MODEL_T_IF" \
    --domains math,code,if \
    --train "$RL_TRAIN" --val "$RL_VAL" --output "$RUNS/dry-mt-opd" --gpus "$GPUS"

run_dry "eval.sh" bash "$REPO/eval.sh" \
    --model "$MODEL_STUDENT" --input "$EVAL_INPUT" \
    --output "$RUNS/dry-eval" --gpus "$GPUS"

banner "25 — negative checks (validation gates that fire only under --run)"
{
    echo "### mt_opd.sh with a single teacher under --run must be rejected"
    set +e
    bash "$REPO/scripts/local/mt_opd.sh" --model "$MODEL_STUDENT" \
        --teacher "$MODEL_T_MATH" --domains math \
        --train "$RL_TRAIN" --val "$RL_VAL" --output "$RUNS/dry-neg" --run 2>&1 | head -5
    echo "exit=${PIPESTATUS[0]} (nonzero expected)"
    echo
    echo "### mt_opd.sh with 3 teachers but 2 domains must be rejected"
    bash "$REPO/scripts/local/mt_opd.sh" --model "$MODEL_STUDENT" \
        --teacher "$MODEL_T_MATH" --teacher "$MODEL_T_CODE" --teacher "$MODEL_T_IF" \
        --domains math,code \
        --train "$RL_TRAIN" --val "$RL_VAL" --output "$RUNS/dry-neg" 2>&1 | head -5
    echo "exit=${PIPESTATUS[0]} (nonzero expected)"
    set -e
} 2>&1 | tee -a "$LOG"

echo "[25] done. log=$LOG"
