## Modification History

| Date       | Summary of Changes                                              |
| ---------- | --------------------------------------------------------------- |
| 2026-09-11 | Add worker-portable persistent-venv rule and one-run preflight contract for Open-MOPD E2E. |
| 2026-09-02 | Record MinerU environment recovery                              |
| 2026-09-02 | Add Open-MOPD / verl 0.7.0.dev env build, `/data` RBD contention, Hydra pre-flight |

### MinerU 3.4.0 on Python 3.13

The available environment is `/data/ycfeng/envs/mineru` with Python `3.13.13` and MinerU `3.4.0`. The original skill path `/local/ycfeng/anaconda3/envs/serena_env/bin/mineru` may be absent.

For low-load CLI startup, make VLM, Office, `config_reader`, and `guess_suffix_or_lang` imports lazy in the installed MinerU CLI modules. This makes `mineru --version` return normally. The local `fast_api -d cpu` pipeline can still fail its health startup under severe host load; capture the explicit timeout and use a validated fallback only when the deliverable requires immediate PDF text and page images.

---

### Open-MOPD / verl 0.7.0.dev: the version set that actually works

`training/install_requirements.sh` installs **no rollout backend**, while all three PPO
launchers hard-code `rollout.name=vllm`. It also never pins torch — torch arrives only
transitively via `accelerate`/`peft`/`torchdata`/`tensordict` (`training/verl/setup.py:26-45`),
so its CUDA build is whatever the default index serves.

Verified 2026-09-02 (Python 3.12.3; target H800, driver 570.x / CUDA compat 12.8):

```bash
uv venv --python 3.12 /data/ycfeng/envs/openmopd-py312
PY=/data/ycfeng/envs/openmopd-py312/bin/python
export UV_CACHE_DIR=/data/ycfeng/tmp/uv-cache        # never /tmp — 16G tmpfs, will OOM
uv pip install --python $PY vllm==0.11.0             # FIRST: it pins torch 2.8.x cu128
uv pip install --python $PY -e /data/ycfeng/Open-MOPD/training/verl
uv pip install --python $PY ray==2.55.1 transformers==4.57.6 protobuf==3.20.3
uv pip install --python $PY -r /data/ycfeng/Open-MOPD/training/requirements-if-rl.txt
uv pip install --python $PY -e /data/ycfeng/Open-MOPD/training/third_party/verifiable-instructions
uv pip install --python $PY -r /data/ycfeng/Open-MOPD/evals/verifier/requirements.txt
uv pip install --python $PY swanlab math-verify duckdb matplotlib pytest
```

Why `vllm==0.11.0`: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:303` declares
`minver = "0.11.0"`, and `verl/third_party/vllm/__init__.py:46` accepts `>= 0.7.0`.

**flash-attn is optional — skip it.** It is needed only by the padding-free FSDP path, so run
with `actor_rollout_ref.model.use_remove_padding=False` (and the same on each
`mt_reward_model_{i}.model.*`). vLLM ships its own `vllm/vllm_flash_attn/`, so the rollout
side is unaffected. Building it from source costs ~30 min.

Ordering trap: verl pins `numpy<2.0.0`, so installing it into a numpy-2 env silently
downgrades numpy.

**Build the venv on `/data`, not on the GPU worker.** An `rlaunch` worker's local rootfs is
wiped on reclaim; `/data/ycfeng/envs/` survives and is auto-mounted on empty-image workers.
Build once on the CPU master — pip works there and no GPU quota is spent — then have the
worker activate it.

### Worker-portable persistent-venv rule (verified 2026-09-11)

Treat a missing library or ABI error observed on a GPU worker as a CPU-host environment
defect. Capture the first traceback, image name, Python path, and Torch/CUDA versions;
then repair the single persistent venv under `/data/ycfeng/envs/` and rerun its import
probe before allocating another worker. The normal worker wrapper must contain no broad
`pip install` or dependency resolution. A worker-side install is permitted only as a
bounded diagnosis; every discovered requirement must be added to the CPU-side recipe
and rebuilt there.

The interpreter itself must be visible from the mount. `uv venv` can create
`bin/python` symlinks through a CPU-only `$HOME/.local` path, which is absent in a GPU
worker. `e2e/00_setup_env.sh` detects that case, copies the CPython runtime into
`$VENV/.python-runtime`, and repoints `bin/python`/`bin/python3` before installing
packages. Verify this property on the CPU host:

```bash
VENV=/data/ycfeng/envs/openmopd-py312
test -x "$VENV/bin/python" && test -x "$VENV/bin/torchrun"
readlink -f "$VENV/bin/python"
"$VENV/bin/python" -c 'import torch, vllm, transformers, ray, verl, verifiable_instructions; print(torch.__version__, torch.version.cuda)'
```

Only after this probe and the `punkt`/`punkt_tab` NLTK data check pass should a GPU RJob
be submitted. The worker then uses absolute paths from the mounted venv, runs the same
preflight again, and starts the workload. This makes retries after worker reclaim
byte-for-byte reusable and avoids spending GPU time discovering one missing package at
a time.

### pip on the CPU master needs no proxy

`http://mirrors.i.basemind.com/pypi/simple/` and `http://pypi.i.basemind.com/brain/dev/+simple`
answer directly. `eval $(curl -s http://deploy.i.shaipower.com/httpproxy)` was not required
for any install in this workspace. Reach for the proxy only after a mirror miss.

### `/data` is Ceph RBD — parallel agent fan-out starves large installs

`df -hT` shows `/data` on `/dev/rbd1` (xfs), not NFS. It is the same device every subagent
reads source from.

Symptom: a `uv pip install` of the CUDA wheel set slowed to **32 MB/min** with
`%Cpu(s): 90.3 wa` and load average **235**, while two agent workflows (29 + 10 agents) ran.

Two causes, both avoidable:

1. Large agent fan-out competing for the same block device — start big installs before or
   after a fan-out, not during.
2. **Polling with `du -sb` on the uv cache** (~300k files) or `ls -la` on `/data/ycfeng/tmp`
   (12 000+ entries). The poll itself was a large share of the iowait.

Poll a background install by grepping its log, never by walking the tree. To see what `uv`
has left, diff its own log lines:

```bash
grep -oE 'Downloading [a-z0-9._-]+' "$LOG" | awk '{print $2}' | sort -u > /data/ycfeng/tmp/r.txt
grep -oE 'Downloaded [a-z0-9._-]+'  "$LOG" | awk '{print $2}' | sort -u > /data/ycfeng/tmp/d.txt
comm -23 /data/ycfeng/tmp/r.txt /data/ycfeng/tmp/d.txt
```

### `conda env list` and HF-cache probes can exceed 120 s

`conda env list` (`/home/i-fengyicheng/miniconda3/bin/conda`) and `find`/`ls` into
`/data/ycfeng/hf_home_*/hub/models--*/snapshots/` both timed out at 120 s on 2026-09-02.
Probe a specific env's python directly instead of enumerating:

```bash
timeout 60 /path/to/env/bin/python -c "import torch, vllm; print(torch.__version__, vllm.__version__)"
```

When a task only needs *a* tokenizer, synthesizing one beats waiting on the cache — see
`task_memory/task_2026-09-02_mopd_arch_e2e/e2e/10_make_dummy_assets.py`, which builds a
ByteLevel-BPE tokenizer plus four tiny Qwen3 models from scratch with zero downloads.

### Catch Hydra key errors before spending GPU time

A wrong Hydra override in a verl run fails only *after* Ray boots, vLLM allocates KV cache,
and every FSDP model loads. Compose with no execution first:

```python
from hydra import compose, initialize_config_dir
with initialize_config_dir(config_dir="<repo>/training/verl/verl/trainer/config", version_base=None):
    cfg = compose(config_name="ppo_trainer", overrides=[...])   # raises here, in ~1 s
```

A ready-made probe covering every Open-MOPD stage:
`task_memory/task_2026-09-02_mopd_arch_e2e/e2e/15_config_probe.py`.

This is how two shipped-launcher defects were pinned: `scripts/local/opd.sh:66` and
`scripts/local/mt_opd.sh:87` pass `actor_rollout_ref.rollout.reward_mode=` without the `+`
that an undeclared key requires, and `log_prob_top_k` is absent from the composed config so
it reads as `0` — which makes `reward_mode=mt_opd` raise on the first step.
