#!/usr/bin/env bash
# Open-MOPD E2E — ordered driver.
#
# Usage:
#   bash run_all.sh            # everything
#   bash run_all.sh cpu        # only the stages that need no GPU
#   bash run_all.sh gpu        # only the stages that need a GPU
#   bash run_all.sh 60         # a single stage by number
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

MODE="${1:-all}"

CPU_STAGES=(20_unit_tests.sh 25_dry_runs.sh)
GPU_STAGES=(30_sft.sh 35_merge.sh 40_rl.sh 50_opd.sh 60_mt_opd.sh 70_eval.sh)

declare -A RESULT

run_stage() {
    local s="$1"
    banner "RUN $s"
    set +e
    bash "$SCRIPT_DIR/$s"
    local rc=$?
    set -e
    RESULT["$s"]=$rc
    echo "[run_all] $s -> exit $rc"
}

if [[ "$MODE" == "all" || "$MODE" == "cpu" || "$MODE" == "gpu" ]]; then
    if [[ ! -d "$ASSETS/models/student" ]]; then
        banner "RUN 10_make_dummy_assets.py"
        "$PY" "$SCRIPT_DIR/10_make_dummy_assets.py" --out "$ASSETS" --clean
    else
        echo "[run_all] assets already present at $ASSETS (delete to rebuild)"
    fi
fi

if [[ "$MODE" == "all" || "$MODE" == "cpu" || "$MODE" == "gpu" ]]; then
    banner "RUN 15_config_probe.py (compose-only; catches Hydra key errors before any GPU work)"
    set +e
    "$PY" "$SCRIPT_DIR/15_config_probe.py" --assets "$ASSETS" 2>&1 | tee "$LOGS/15_config_probe.log"
    RESULT[15_config_probe.py]=${PIPESTATUS[0]}
    set -e
fi

case "$MODE" in
    all) for s in "${CPU_STAGES[@]}" "${GPU_STAGES[@]}"; do run_stage "$s"; done ;;
    cpu) for s in "${CPU_STAGES[@]}"; do run_stage "$s"; done ;;
    gpu) for s in "${GPU_STAGES[@]}"; do run_stage "$s"; done ;;
    *)   match=$(ls "$SCRIPT_DIR" | grep -E "^${MODE}_" | head -1)
         [[ -n "$match" ]] || { echo "no stage matching '$MODE'" >&2; exit 2; }
         run_stage "$match" ;;
esac

banner "SUMMARY"
fail=0
for s in "${!RESULT[@]}"; do
    printf '  %-24s %s\n' "$s" "$([[ ${RESULT[$s]} -eq 0 ]] && echo PASS || echo "FAIL(${RESULT[$s]})")"
    [[ ${RESULT[$s]} -eq 0 ]] || fail=1
done
echo "[run_all] logs under $LOGS"
exit $fail
