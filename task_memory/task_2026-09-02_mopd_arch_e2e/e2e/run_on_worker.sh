#!/usr/bin/env bash
# Open-MOPD E2E — the single command an rlaunch GPU worker runs.
#
# Everything it needs is already on /data (persistent): the venv, the dummy assets, and
# these scripts. The worker contributes only GPUs. Logs land back on /data so a reclaim
# does not lose them.
#
# Launch from the CPU master with:
#   rlaunch --charged-group=codesign --private-machine=group --positive-tags=h800 \
#           --gpu=1 --cpu=16 --memory=131072 --backoff-limit=1 \
#           -- bash /data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e/e2e/run_on_worker.sh
set -uo pipefail          # NOT -e: every stage's exit code is recorded, not fatal

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

banner "WORKER — environment"
echo "host: $(hostname)"
nvidia-smi -L || echo "  no nvidia-smi (NOT a GPU worker — stop here)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| devices', torch.cuda.device_count())"
echo "assets: $ASSETS"
"$PY" -c "import json;print(json.dumps(json.load(open('$ASSETS/manifest.json')), indent=2))" 2>/dev/null || echo "  no manifest"

declare -A RC

stage() {
    local s="$1"
    banner "WORKER — $s"
    local t0=$SECONDS
    # Keep the chain running so later stages can produce their own evidence.
    set +e
    bash "$SCRIPT_DIR/$s"
    local rc=$?
    set -e
    RC["$s"]=$rc
    echo "[worker] $s -> exit ${RC[$s]}  ($((SECONDS-t0))s)"
}

# CPU-only first: they are cheap and they gate the GPU stages.
stage 20_unit_tests.sh
stage 25_dry_runs.sh
banner "WORKER — 15_config_probe.py"
"$PY" "$SCRIPT_DIR/15_config_probe.py" --assets "$ASSETS" 2>&1 | tee "$LOGS/15_config_probe.log"
RC[15_config_probe.py]=${PIPESTATUS[0]}

# GPU stages, in recipe order.
stage 30_sft.sh
stage 35_merge.sh
stage 40_rl.sh
stage 50_opd.sh

# MT-OPD twice: naive (what the release gives you) then M1 (the paper's fix), so the
# per-domain loss_weight difference is visible in one run.
banner "WORKER — 60_mt_opd.sh (MT_MODE=naive)"
set +e
MT_MODE=naive bash "$SCRIPT_DIR/60_mt_opd.sh"; RC["60_mt_opd:naive"]=$?
set -e
echo "[worker] 60_mt_opd:naive -> exit ${RC[60_mt_opd:naive]}"

banner "WORKER — 60_mt_opd.sh (MT_MODE=m1)"
set +e
MT_MODE=m1 bash "$SCRIPT_DIR/60_mt_opd.sh"; RC["60_mt_opd:m1"]=$?
set -e
echo "[worker] 60_mt_opd:m1 -> exit ${RC[60_mt_opd:m1]}"

stage 70_eval.sh

banner "WORKER — SUMMARY"
fail=0
for s in "${!RC[@]}"; do
    printf '  %-28s %s\n' "$s" "$([[ ${RC[$s]} -eq 0 ]] && echo PASS || echo "FAIL(${RC[$s]})")"
    [[ ${RC[$s]} -eq 0 ]] || fail=1
done
echo
echo "[worker] logs: $LOGS"
echo "[worker] runs: $RUNS"
exit "$fail"
