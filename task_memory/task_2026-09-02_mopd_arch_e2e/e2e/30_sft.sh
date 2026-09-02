#!/usr/bin/env bash
# Open-MOPD E2E — Step 30: MixSFT stage, driven through the released launcher.
#
# Entrypoint: scripts/local/sft.sh --run  ->  torchrun -m verl.trainer.sft_trainer
#
# Schema note that costs an hour if you miss it: verl/trainer/sft_trainer.py:378 is
#   @hydra.main(config_name="sft_trainer_engine")
# so verl/trainer/config/sft_trainer.yaml (prompt_key=question / response_key=answer)
# belongs to the OTHER trainer, fsdp_sft_trainer.py, and is not used here.
# sft_trainer.py:393 always constructs MultiTurnSFTDataset, which reads the column
# named by config["multiturn"]["messages_key"] (multiturn_sft_dataset.py:63-66).
# sft_trainer_engine.yaml declares messages_key flat under `data:` (line 26), which that
# nested lookup never reaches, so the live column name is the hard default "messages".
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
require_assets

OUT="$RUNS/sft"
CKPT="$OUT/checkpoints"
LOG="$LOGS/30_sft.log"
mkdir -p "$CKPT"

banner "30 — MixSFT via scripts/local/sft.sh --run"
set +e
env PYTHON_BIN="$PY" TORCHRUN_BIN="$TORCHRUN" \
    TRAIN_BATCH_SIZE=4 MAX_LENGTH=256 TOTAL_EPOCHS=1 RESUME_MODE=disable \
    PROJECT_NAME=OpenMOPD-e2e EXPERIMENT_NAME=sft-dummy \
    bash "$REPO/scripts/local/sft.sh" --run \
        --model "$MODEL_STUDENT" \
        --train "$SFT_TRAIN" --val "$SFT_VAL" \
        --output "$OUT" --checkpoint "$CKPT" --gpus "$GPUS" \
        --extra "data.micro_batch_size_per_gpu=2" \
        --extra "data.max_token_len_per_gpu=1024" \
        --extra "+model.override_config.attn_implementation=eager" \
        --extra "optim.lr=1e-4" \
        --extra "trainer.save_freq=1" \
        --extra "trainer.test_freq=1" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

banner "30 — evidence"
echo "[30] exit=$rc"
echo "[30] loss / step lines:"
grep -iE "loss|step" "$LOG" | tail -25 || true
echo "[30] checkpoint tree:"
find "$CKPT" -maxdepth 3 2>/dev/null | head -40 || true
echo "[30] latest_checkpointed_iteration.txt:"
cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo "  (absent)"
exit $rc
