#!/usr/bin/env bash
# Open-MOPD E2E — Step 00: build the NFS-persistent Python environment.
#
# Runs on ANY node (CPU master or GPU worker). The venv lives on the workspace
# NFS so a reclaimed GPU worker does not cost a reinstall.
#
# Version choices and their evidence:
#   vllm==0.11.0      training/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:303
#                     declares minver = "0.11.0" for the sleep-level API, and
#                     verl/third_party/vllm/__init__.py:46 accepts >= 0.7.0.
#   ray==2.55.1       training/install_requirements.sh
#   transformers==4.57.6  training/install_requirements.sh
#   protobuf==3.20.3  training/install_requirements.sh
#   flash-attn        DELIBERATELY OMITTED. See notes in 60_mt_opd.sh: the dummy
#                     model runs with use_remove_padding=False, which is the only
#                     path that needs it. Building it takes ~30 min.
set -euo pipefail

REPO="${REPO:-/data/ycfeng/Open-MOPD}"
VENV="${VENV:-/data/ycfeng/envs/openmopd-py312}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/ycfeng/tmp/uv-cache}"
export TMPDIR="${TMPDIR:-/data/ycfeng/tmp}"
export NLTK_DATA="${NLTK_DATA:-$VENV/nltk_data}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
mkdir -p "$NLTK_DATA"

echo "[00] repo=$REPO venv=$VENV"

if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv --python 3.12 "$VENV"
fi

PY="$VENV/bin/python"
# uv-managed interpreters are normally symlinked through the CPU host's
# ~/.local tree, which is not visible inside a GPU worker. Copy that small
# CPython runtime into the mounted /data tree and repoint the venv launcher.
PY_REAL="$(readlink -f "$PY")"
if [[ "$PY_REAL" != "$VENV"/* ]]; then
    PY_ROOT="$(cd "$(dirname "$PY_REAL")/.." && pwd)"
    RUNTIME_ROOT="$VENV/.python-runtime"
    if [[ ! -x "$RUNTIME_ROOT/bin/python3.12" ]]; then
        echo "[00] copying relocatable CPython runtime to $RUNTIME_ROOT"
        cp -a "$PY_ROOT" "$RUNTIME_ROOT"
    fi
    ln -sfn "$RUNTIME_ROOT/bin/python3.12" "$VENV/bin/python"
    ln -sfn "$RUNTIME_ROOT/bin/python3.12" "$VENV/bin/python3"
    PY="$VENV/bin/python"
fi
pipi() { uv pip install --python "$PY" "$@"; }

echo "[00] 1/7 torch + vllm (pulls torch 2.8.x cu128, matches the H800 570.x driver)"
pipi "vllm==0.11.0"

echo "[00] 2/7 verl editable (brings hydra, tensordict, codetiming, math_verify, ...)"
pipi -e "$REPO/training/verl"

echo "[00] 3/7 pinned runtime versions from training/install_requirements.sh"
pipi "ray==2.55.1" "transformers==4.57.6" "protobuf==3.20.3"

echo "[00] 4/7 instruction-following RL deps"
pipi -r "$REPO/training/requirements-if-rl.txt"
pipi -e "$REPO/training/third_party/verifiable-instructions"

echo "[00] 5/7 verifier deps"
pipi -r "$REPO/evals/verifier/requirements.txt"

echo "[00] 6/7 misc training deps"
pipi swanlab math-verify duckdb matplotlib pytest
# vllm imports numba during engine startup. Keep NumPy within numba's supported
# range while remaining compatible with the vllm/scipy wheels above.
pipi "numpy==2.2.6"
# The misc tools may upgrade protobuf; restore the version pinned by
# training/install_requirements.sh for the final runtime contract.
pipi "protobuf==3.20.3"

echo "[00] 7/7 nltk data (punkt) — needed by the IF verifier"
NLTK_ALLOW_PROXIED_URLOPEN=1 "$PY" - <<'PY' || echo "[00] nltk download failed; IF scoring may degrade"
import nltk
for r in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{r}")
    except LookupError:
        nltk.download(r, quiet=True)
PY

echo "[00] === version report ==="
"$PY" - <<'PY'
import importlib
for m in ("torch", "vllm", "transformers", "ray", "numpy", "tensordict",
          "hydra", "omegaconf", "pyarrow", "pandas", "verl", "tokenizers",
          "safetensors", "nltk", "verifiable_instructions"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:24s} {getattr(mod, '__version__', 'n/a')}")
    except Exception as e:
        print(f"  {m:24s} MISSING ({type(e).__name__})")
import torch
print(f"  cuda available           {torch.cuda.is_available()}")
print(f"  torch cuda build         {torch.version.cuda}")
PY
echo "[00] done. activate with: source $VENV/bin/activate"
