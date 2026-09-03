# Double-Check Test Report — Future Work

## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-04 | Recorded direct verification for the five follow-up items and the scoped repairs. |

## 1. Test Script Information

Test scripts and source checks:

* `/data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e/e2e/15_config_probe.py`
* `/data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e/e2e/25_dry_runs.sh`
* `/data/ycfeng/Open-MOPD/experiments/tests/test_if_rl_grm.py`
* `/data/ycfeng/Open-MOPD/experiments/tests/test_capability_subspace.py`
* `/data/ycfeng/Open-MOPD/training/verl/tests/test_mt_opd_m2_m3.py`
* `/data/ycfeng/Open-MOPD/training/verl/tests/test_mt_opd_domain_metrics.py`
* `/data/ycfeng/Open-MOPD/training/verl/tests/trainer/ppo/test_mt_opd_batch_routing.py`

Reproducible commands:

```bash
cd /data/ycfeng/Open-MOPD
PYTHONPATH=training/verl /data/ycfeng/envs/openmopd-py312/bin/python \
  task_memory/task_2026-09-02_mopd_arch_e2e/e2e/15_config_probe.py \
  --assets /data/ycfeng/tmp/openmopd-e2e-assets-eager2-20260903

PYTHONPATH=training/verl /data/ycfeng/envs/openmopd-py312/bin/python -m pytest -q \
  experiments/tests/test_if_rl_grm.py

PYTHONPATH=training/verl /data/ycfeng/envs/openmopd-py312/bin/python -m pytest -q \
  experiments/tests/test_capability_subspace.py

PYTHONPATH=training/verl /data/ycfeng/envs/openmopd-py312/bin/python -m pytest -q \
  training/verl/tests/test_mt_opd_m2_m3.py \
  training/verl/tests/test_mt_opd_domain_metrics.py \
  training/verl/tests/trainer/ppo/test_mt_opd_batch_routing.py

ASSETS=/data/ycfeng/tmp/openmopd-e2e-assets-eager2-20260903 \
PYTHON_BIN=/data/ycfeng/envs/openmopd-py312/bin/python \
  bash task_memory/task_2026-09-02_mopd_arch_e2e/e2e/25_dry_runs.sh
```

Environment: conda/venv `openmopd-py312`, Python `3.12.3`, Torch `2.8.0+cu128`.

## 2. Validation Criteria

* Launcher argv contains `+actor_rollout_ref.rollout.reward_mode` and
  `+actor_rollout_ref.rollout.log_prob_top_k=256`.
* Hydra probe reports `unexpected outcomes: 0`; the plain undeclared `reward_mode` case
  remains a deliberate `RAISE` negative regression.
* MT-OPD binding accepts three unique non-empty domains with two additional teachers and
  rejects count mismatch, duplicates, and empty labels.
* GRM tests exercise the concurrent fan-out transport seam without DNS access or retry
  delays; FakeSession satisfies `_get_session()`'s `.mount()` contract.
* Explicit capability directions are deterministic and agree with exact SVD retained
  energy within the existing `0.02` absolute tolerance.
* MT-OPD policy and conflict metric use the configured `conflict_nats` value.
* M2 default direction and `dummy_math` scorer remain unchanged unless a separate behavior
  decision authorizes them.

## 3. Test Results and Evidence

| Target | Result | Observed evidence |
| ------ | ------ | ----------------- |
| Hydra composition | PASS | 11 cases; `unexpected outcomes: 0`; invalid plain forms `RAISE`, launcher forms `COMPOSE` |
| Launcher dry-run | PASS | OPD argv includes `+reward_mode=opd_kl` and `+log_prob_top_k=256`; MT-OPD argv includes `+reward_mode=mt_opd`, `+log_prob_top_k=256`, teacher-0 `input_tokenizer=null`, `use_remove_padding=True`, `param_offload=True` |
| MT-OPD routing/M2/M3 | PASS | `60 passed` (one Ray deprecation warning); conflict tests cover `1.0`-nat threshold and policy masking |
| GRM | PASS | `17 passed in 4.82s`; no `fake-endpoint` DNS requests after transport stubbing |
| Capability subspace | PASS | `69 passed in 5.09s`; deterministic direction regression and retained-energy comparison pass |

The M2 default decision is intentionally deferred. Existing source evidence records
`divide` as historically harmful and tests demonstrate explicit `multiply` behavior, but
changing the default alters optimization semantics for existing runs. The `dummy_math`
scorer is also intentionally deferred because no production registry entry exists and the
current offline eval correctly reports `unknown_dataset` with `scored_rows=0`.
