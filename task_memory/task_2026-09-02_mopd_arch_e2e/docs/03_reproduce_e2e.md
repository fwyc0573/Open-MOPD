# Open-MOPD E2E Reproduction Guide

## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-03 | Added the verified H800 reproduction procedure, offline dummy assets, stage commands, evidence locations, and known release-test failures. |
| 2026-09-04 | Updated OPD/MT-OPD launcher commands after repairing Hydra extension keys and teacher-0 settings. |

This guide reproduces the high-fidelity single-GPU flow used for this task. It uses
four locally synthesized `Qwen3ForCausalLM` models and never downloads Hugging Face
weights. The worker writes all logs and outputs below `/data/ycfeng/tmp`, which is an
NFS-mounted path on the GPU worker.

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
  --output "$ASSETS"
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
