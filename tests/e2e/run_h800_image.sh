#!/usr/bin/env bash
# Run the existing offline E2E chain in the verified company H800 image.
set -euo pipefail
export REPO=/data/ycfeng/Open-MOPD
: "${WORK:?A fresh persistent output directory is required}"
export ASSETS="$WORK/assets" LOGS="$WORK/logs" RUNS="$WORK/runs"
export TMPDIR=/data/ycfeng/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$WORK/hf-cache" PYTHONUNBUFFERED=1
# The Python environment is built once on the CPU host and exposed through
# the local /data mount. Reusing it keeps worker startup deterministic and
# avoids dependency resolution or pip installs on ephemeral GPU workers.
export VENV="${VENV:-/data/ycfeng/envs/openmopd-py312}"
export NLTK_DATA="${NLTK_DATA:-$VENV/nltk_data}"
export TORCHRUN="${TORCHRUN:-$VENV/bin/torchrun}"
export PYTHONPATH="$REPO/training/verl:$REPO:${PYTHONPATH:-}"
mkdir -p "$LOGS"
exec > >(tee "$LOGS/worker.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$WORK/exit_code"' EXIT
PY="$VENV/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "[h800] persistent Python environment is missing: $PY" >&2
    echo "[h800] build it on the CPU host with e2e/00_setup_env.sh before submitting a worker" >&2
    exit 127
fi
if [[ ! -x "$TORCHRUN" ]]; then
    echo "[h800] persistent torchrun executable is missing: $TORCHRUN" >&2
    exit 127
fi
$NVIDIA_SMI -L
"$PY" --version
printf 'Python=%s Torchrun=%s\n' "$PY" "$TORCHRUN"
"$PY" - <<'PY'
import importlib
import torch

required = ("vllm", "transformers", "ray", "tensordict", "hydra", "verl",
            "verifiable_instructions")
versions = {}
for name in required:
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "n/a")
print("runtime_versions", versions)
print("torch", torch.__version__, "cuda_build", torch.version.cuda,
      "cuda_available", torch.cuda.is_available())
PY
"$PY" "$REPO/task_memory/task_2026-09-02_mopd_arch_e2e/e2e/10_make_dummy_assets.py" --out "$ASSETS"
bash "$REPO/task_memory/task_2026-09-02_mopd_arch_e2e/e2e/run_on_worker.sh"
