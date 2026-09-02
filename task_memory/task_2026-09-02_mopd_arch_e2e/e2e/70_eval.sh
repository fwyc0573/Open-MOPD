#!/usr/bin/env bash
# Open-MOPD E2E — Step 70: evaluation layer.
#
# Two independent stages (evals/README.md):
#   1. offline vLLM rollout   -> evals/rollout_engine/vllm_rollout.py
#   2. scoring / aggregation  -> evals/score_rollouts.py and evals/verifier/score.py
#
# Stage 1 needs a GPU. Stage 2 is pure CPU and can be re-run on any node against the
# rollout parquet produced by stage 1.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
require_assets

# Which model to evaluate. Defaults to the MT-OPD student checkpoint if the merged
# HF export exists, otherwise the dummy student.
EVAL_MODEL="${EVAL_MODEL:-$MODEL_STUDENT}"
OUT="$RUNS/eval"
LOG="$LOGS/70_eval.log"
mkdir -p "$OUT"

banner "70a — offline vLLM rollout"
cd "$REPO"
set +e
"$PY" -m evals.rollout_engine.vllm_rollout \
    --model "$EVAL_MODEL" \
    --input "$EVAL_INPUT" \
    --output-dir "$OUT" \
    --temperature 0.6 --top-p 0.95 --top-k 20 \
    --n 2 \
    --max-tokens "$MAX_RESPONSE_LEN" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.30 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --dtype bfloat16 2>&1 | tee "$LOG"
rc1=${PIPESTATUS[0]}
set -e

echo "[70a] exit=$rc1"
echo "[70a] rollout parquet:"; ls -la "$OUT"/*.parquet 2>/dev/null || echo "  (none)"

banner "70a — rollout parquet schema + a sample completion"
"$PY" - "$OUT" <<'PY' 2>&1 | tee -a "$LOG"
import sys, glob, pandas as pd
files = sorted(glob.glob(f"{sys.argv[1]}/*.parquet"))
if not files:
    print("  no parquet produced"); raise SystemExit(0)
df = pd.read_parquet(files[0])
print("  file:", files[0])
print("  rows:", len(df))
print("  columns:", list(df.columns))
if len(df):
    r = df.iloc[0]
    print("  completion[0] repr:", repr(str(r.get('completion',''))[:160]))
    print("  completion_tokens:", r.get('completion_tokens'))
    print("  finish_reason:", r.get('finish_reason'))
PY

banner "70b — score_rollouts (CPU)"
set +e
"$PY" -m evals.score_rollouts \
    --rollout-dir "$OUT" \
    --dataset dummy_math \
    --out "$OUT/scores.json" 2>&1 | tee -a "$LOG"
rc2=${PIPESTATUS[0]}
set -e
echo "[70b] exit=$rc2"
[[ -f "$OUT/scores.json" ]] && { echo "[70b] scores.json:"; cat "$OUT/scores.json"; }

banner "70c — verifier/score (CPU, the official-vs-live rate path)"
set +e
FIRST=$(ls "$OUT"/*_rollouts_*.parquet 2>/dev/null | head -1)
if [[ -n "$FIRST" ]]; then
    "$PY" -m evals.verifier.score --rollout "$FIRST" --output "$OUT/verifier_score.json" 2>&1 | tee -a "$LOG"
    rc3=${PIPESTATUS[0]}
    [[ -f "$OUT/verifier_score.json" ]] && cat "$OUT/verifier_score.json"
else
    echo "[70c] no rollout parquet to score"; rc3=1
fi
set -e
echo "[70c] exit=$rc3"

exit $(( rc1 != 0 || rc2 != 0 ))
