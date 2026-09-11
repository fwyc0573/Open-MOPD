# Open-MOPD E2E Reproduction Guide

## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-11 | Make the persistent CPU-built environment and one-command H800 rerun the authoritative procedure; add preflight and reuse checks. |
| 2026-09-10 | Add current local Python submission and observation procedure; mark historical CLI recipes and correct generator argument. |
| 2026-09-03 | Added the verified H800 reproduction procedure, offline dummy assets, stage commands, evidence locations, and known release-test failures. |
| 2026-09-04 | Updated OPD/MT-OPD launcher commands after repairing Hydra extension keys and teacher-0 settings. |

This guide reproduces the high-fidelity single-GPU flow used for this task. It uses
four locally synthesized `Qwen3ForCausalLM` models and never downloads Hugging Face
weights. The worker writes all logs and outputs below `/data/ycfeng/tmp`, which is an
NFS-mounted path on the GPU worker.

## Recommended repeat run: prepare once on the CPU host, submit once to H800

The stable execution contract is a persistent environment on `/data`, followed by one
StepMind Python submission. Build or repair the environment on the CPU host before
requesting a GPU; do not install packages in the worker. The worker image supplies the
CUDA driver and vLLM system integration, while `/data/ycfeng/envs/openmopd-py312`
supplies the complete Python runtime and package set. This removes dependency solving
from the critical GPU allocation window and allows a reclaimed worker to be retried with
the same bytes.

Run this preparation block from `/data/ycfeng/Open-MOPD` whenever the environment is
missing or its version probe fails (normally only once):

```bash
export REPO=/data/ycfeng/Open-MOPD
export VENV=/data/ycfeng/envs/openmopd-py312
export UV_CACHE_DIR=/data/ycfeng/tmp/uv-cache
export TMPDIR=/data/ycfeng/tmp
export UV_INDEX_URL=https://artifactory.stepfun-inc.com/artifactory/api/pypi/pypi-public/simple/
timeout 1200s bash "$REPO/task_memory/task_2026-09-02_mopd_arch_e2e/e2e/00_setup_env.sh"

"$VENV/bin/python" - <<'PY'
import importlib
import nltk
required = ("torch", "vllm", "transformers", "ray", "tensordict", "hydra",
            "omegaconf", "pyarrow", "pandas", "verl", "verifiable_instructions")
for name in required:
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "n/a"))
for resource in ("punkt", "punkt_tab"):
    nltk.data.find(f"tokenizers/{resource}")
import torch
assert torch.__version__.startswith("2.8."), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
print("torch_cuda_build", torch.version.cuda)
PY
```

The import probe must complete without `ModuleNotFoundError`; `punkt` and `punkt_tab`
must be present under `$VENV/nltk_data`. The expected pinned values are Torch
`2.8.x+cu128`, vLLM `0.11.0`, Transformers `4.57.6`, and Ray `2.55.1`. The probe runs
on CPU and does not require a GPU. If it fails, repair the same venv and rerun the probe;
do not add a one-off `pip install` to the worker command.

After the probe passes, submit exactly one H800 job and keep the local launcher alive
until it prints `FINAL_STATUS`:

```bash
export STEPMIND_BACKEND=rjob
"$STEPMIND_PYTHON" -u /data/ycfeng/Open-MOPD/tests/e2e/submit_h800_image.py
```

The launcher creates a fresh `/data/ycfeng/tmp/openmopd-image-<timestamp>` directory,
mounts local `/data/ycfeng` at the same path in the worker, and runs
`tests/e2e/run_h800_image.sh`. That wrapper validates the mounted interpreter and all
critical imports before generating dummy assets or starting training. A missing
`$VENV/bin/python` or `$VENV/bin/torchrun` fails immediately with an actionable message,
which means the environment can be repaired on CPU without consuming another long GPU
attempt. Do not submit a duplicate while the first RJob is queued; inspect its printed
ID with the runbook controller instead.

For a successful rerun, inspect the worker's `logs/worker.log`, per-stage logs, and
`exit_code` under the printed work directory. The acceptance chain is: config probe,
unit tests, SFT, merge, RL, corrected OPD, corrected MT-OPD (`naive` and `m1`), and
vLLM rollout/eval. The wrapper preserves every stage exit code in its summary so an
apparently successful process exit is not treated as an E2E pass unless all required
stage artifacts are present.

## Current machine: personal Python H800 submission (2026-09-11)

On `kun-workspace-vgen2`, follow [the mandatory Python RJobBackend runbook](/data/ycfeng/stepfun-env-handbook/stepmind-python-rjob.md), sections 2–3, to initialize personal `i-fengyicheng` credentials and proxy settings before creating a client. Use `STEPMIND_BACKEND=rjob`; local `rlaunch`, `brainctl`, and standalone `rjob` submission are prohibited. The one-command procedure above is authoritative.

The verified submission uses `spawn_tasks()`, H800/codesign, one GPU, 16 CPUs, `mem_gb=132`, the explicit company image, and a local `/data/ycfeng` source mount. RJob `exp-0911-152629-174180` completed with terminal status `Succeeded`; its persistent-venv report is [test_report_2026-09-11_personal_h800_persistent_venv.md](../test_report_2026-09-11_personal_h800_persistent_venv.md). Keep the launcher process alive until it prints `FINAL_STATUS`, because the backend exit handler can stop the worker.

For a queued or running job, inspect the same ID with the runbook's Python controller. The initial `Job Create` line contains the ID; do not submit a duplicate while that job is waiting for quota. Evidence is retained under the printed `/data/ycfeng/tmp/openmopd-image-<timestamp>` directory: `logs/worker.log`, per-stage logs, `runs/`, and `exit_code`.

H800 uses `codesign` plus `H800`. H200 quota groups and image recipes are separate routes and are not part of this reproduction.

## Historical r5 procedure (reference only)

The remaining sections record the original environment and r5 observations. On the current machine use the Python procedure above for submission and inspection; do not run these historical local `rlaunch` commands.

## 1. Prerequisites

Run the commands from `/data/ycfeng/Open-MOPD` on a CPU master with `/kubebrain/rlaunch`.
The verified allocation is one H800, 16 CPUs, and 131072 MiB memory:

```bash
rlaunch --predict-only \
  --charged-group=codesign --private-machine=group --positive-tags=h800 \
  --gpu=1 --cpu=16 --memory=131072 --backoff-limit=1 -- bash -lc 'true'
```

The task uses Python 3.12 from `/data/ycfeng/envs/openmopd-py312`. The worker setup
script can create the environment and install the vendored package/dependencies:

```bash
bash task_memory/task_2026-09-02_mopd_arch_e2e/e2e/00_setup_env.sh
```

The final run used `torch 2.8.0+cu128`, CUDA available, one H800, `transformers 4.57.6`,
`vllm 0.11.0`, and Ray. FlashAttention and FlashInfer are intentionally absent.

## 2. Create the offline dummy assets

Generate the assets on the persistent filesystem. The generator creates a byte-level
tokenizer, one student and three seed-distinct teachers (each 441,216 parameters),
and the SFT/RL/eval parquet files:

```bash
ASSETS=/data/ycfeng/tmp/openmopd-e2e-assets-eager2-20260903
mkdir -p /data/ycfeng/tmp
/data/ycfeng/envs/openmopd-py312/bin/python \
  task_memory/task_2026-09-02_mopd_arch_e2e/e2e/10_make_dummy_assets.py \
  --out "$ASSETS"
```

Set offline switches before every model load:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

The expected manifest has tokenizer vocabulary `568`, model parameter count `441216`,
student seed `0`, and teacher seeds `1`, `2`, and `3`. No released checkpoint is used.

## 3. Run the complete worker flow

Use a fresh persistent output directory for each run. `run_on_worker.sh` records every
stage independently and continues after a failure so later evidence remains available.

```bash
rlaunch \
  --name openmopd-e2e-full-20260903-r5 \
  --charged-group=codesign --private-machine=group --positive-tags=h800 \
  --gpu=1 --cpu=16 --memory=131072 --backoff-limit=1 \
  --max-wait-duration=30m -- bash -lc '
    export WORK=/data/ycfeng/tmp/openmopd-e2e-r5-20260903
    export ASSETS=/data/ycfeng/tmp/openmopd-e2e-assets-eager2-20260903
    export REPO=/data/ycfeng/Open-MOPD
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    bash "$REPO/task_memory/task_2026-09-02_mopd_arch_e2e/e2e/run_on_worker.sh"
  '
```

The driver order is:

```text
20_unit_tests.sh -> 25_dry_runs.sh -> 15_config_probe.py -> 30_sft.sh
-> 35_merge.sh -> 40_rl.sh -> 50_opd.sh
-> 60_mt_opd.sh (MT_MODE=naive) -> 60_mt_opd.sh (MT_MODE=m1) -> 70_eval.sh
```

The OPD/MT-OPD launchers now append the undeclared Hydra keys with `+`, set
`log_prob_top_k=256`, and align teacher-0 tokenizer/remove-padding/offload settings.
The E2E scripts retain a plain `reward_mode` composition negative case so the original
Hydra failure remains regression evidence without treating the repaired launcher as broken.

## 4. Inspect evidence

Logs and artifacts are under:

```text
/data/ycfeng/tmp/openmopd-e2e-r5-20260903/logs/
/data/ycfeng/tmp/openmopd-e2e-r5-20260903/runs/
```

Useful commands:

```bash
rg -n "passed|failed|exit=|step:[12]|loss_weight|prompt_share|token_share|conflict|rows:|unknown_dataset" \
  /data/ycfeng/tmp/openmopd-e2e-r5-20260903/logs
cat /data/ycfeng/tmp/openmopd-e2e-r5-20260903/runs/sft/checkpoints/latest_checkpointed_iteration.txt
find /data/ycfeng/tmp/openmopd-e2e-r5-20260903/runs/eval -maxdepth 1 -type f -ls
```

Expected verified observations from r5:

| Stage | Evidence |
| ----- | -------- |
| Open-MOPD tests | `132 passed` in `20_unit_tests.log` |
| SFT | 6 steps; train loss `6.4359403 -> 5.8885908`; validation loss `6.3142653 -> 5.7854004`; checkpoint tracker `6` |
| Merge | `model.safetensors` and tokenizer/config written under `runs/merged/global_step_6` |
| RL | 2 steps; `actor/entropy` `6.3163104`, `6.3164210`; throughput `397.2633`, `695.6351` tokens/s |
| OPD | shipped exit `1` with Hydra `ConfigCompositionException`; corrected exit `0`; `teacher/entropy` `6.3164754`, `6.3166409`; reward means `-0.5174221`, `-0.4962008` |
| MT-OPD naive | corrected exit `0`; all three domains present; conflict spread mean `0.3834707` / `0.3818398` |
| MT-OPD M1 | corrected exit `0`; step-2 token shares code/if/math `0.2920228/0.3511396/0.3568376`; loss weights `1.3697561/0.5695741/1.1209581` |
| Eval | vLLM exit `0`; 12 rollout rows, `completion_tokens=64`; verifier exit `0`, `dummy_math` marked `unknown_dataset` because no scorer is shipped |

## 5. Known non-core test failures

The worker summary is non-zero because the optional repo-level suite contains 8 failures
(`111 passed, 8 failed`). They do not enter the MOPD training/eval chain:

* `experiments/tests/test_if_rl_grm.py` uses a fake `requests.Session` without
  `.mount()` in two tests, and uses the intentionally unresolvable
  `http://fake-endpoint:8000` in six GRM tests. The implementation retries three times
  and reports `NameResolutionError`, so the expected GRM fields remain zero.
* `experiments/tests/test_capability_subspace.py::test_top_rank_retained_matches_the_explicit_projection`
  observes `0.49568635` versus `0.46986032`, exceeding its absolute tolerance `0.02`.

The full logs remain in `20b_repo_tests.log`; these failures are reported rather than
silently suppressed.

## 6. Reproducing one corrected MT-OPD step only

To rerun the paper's central measurement without the whole chain, invoke the existing
stage script with the same environment and an existing dummy asset directory:

```bash
export WORK=/data/ycfeng/tmp/openmopd-mt-opd-repeat-20260903
export ASSETS=/data/ycfeng/tmp/openmopd-e2e-assets-eager2-20260903
export REPO=/data/ycfeng/Open-MOPD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$WORK"
MT_MODE=m1 bash "$REPO/task_memory/task_2026-09-02_mopd_arch_e2e/e2e/60_mt_opd.sh"
```

This command requires the same H800 worker allocation. It writes the per-domain
`prompt_share`, `token_share`, `loss_weight`, reward magnitude, and conflict metrics to
the stage log.
