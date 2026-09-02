#!/usr/bin/env bash
# Open-MOPD E2E — Step 35: merge a verl FSDP checkpoint into a loadable HF dir.
#
# This step exists because stage N+1 CANNOT read stage N's output directly. A verl
# checkpoint under global_step_N/ holds FSDP shards (model_world_size_{WS}_rank_{R}.pt)
# plus a huggingface/ subdir carrying config + tokenizer but NO weights.
#
# The repo ships the wrapper: training/scripts/sft/merge_model.sh, which requires
# fsdp_config.json in the source dir (merge_model.sh:34) and calls
#   python -m verl.model_merger merge --backend fsdp --local_dir X --target_dir Y
# Neither README.md nor scripts/local/README.md mentions this step. It is mandatory.
#
# Usage: 35_merge.sh [SRC_CHECKPOINT_DIR] [DST_DIR]
#   SFT   checkpoints land at  <ckpt>/global_step_N/
#   PPO   checkpoints land at  <ckpt>/global_step_N/actor/   <-- one level deeper
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

SRC="${1:-}"
if [[ -z "$SRC" ]]; then
    SRC=$(find "$RUNS/sft/checkpoints" -maxdepth 1 -name 'global_step_*' -type d 2>/dev/null | sort -V | tail -1)
fi
DST="${2:-$RUNS/merged/$(basename "${SRC:-none}")}"
LOG="$LOGS/35_merge.log"

[[ -n "$SRC" && -d "$SRC" ]] || { echo "[35] no source checkpoint found; pass one explicitly" >&2; exit 2; }

banner "35 — merge_model.sh: $SRC -> $DST"
echo "[35] source tree:"; find "$SRC" -maxdepth 2 | head -25 || true

set +e
env PYTHON_BIN="$PY" TRUST_REMOTE_CODE=1 \
    bash "$REPO/training/scripts/sft/merge_model.sh" "$SRC" "$DST" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

banner "35 — evidence"
echo "[35] exit=$rc"
echo "[35] merged dir:"; ls -la "$DST" 2>/dev/null | head -20 || true
echo "[35] loadable? (config.json + weights present)"
[[ -f "$DST/config.json" ]] && echo "  config.json OK" || echo "  config.json MISSING"
ls "$DST"/*.safetensors "$DST"/*.bin 2>/dev/null | head -3 || echo "  no weight file found"
exit $rc
