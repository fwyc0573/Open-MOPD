# Operational Notes

## Modification History

| Date       | Summary of Changes                          |
| ---------- | ------------------------------------------- |
| 2026-09-02 | Initial environment + repo scouting results |

## Host environment (scouted 2026-09-02)

| Item | Observed value |
| --- | --- |
| `nvidia-smi` | absent → **CPU master node, no GPU** |
| `python3` | `/usr/bin/python3`, Python 3.12.3 |
| `torch` | 2.5.1+cu124, `torch.cuda.is_available() == False` |
| `transformers` / `vllm` / `ray` | **not installed** in system python |
| `verl` | **not importable** (`ModuleNotFoundError: No module named 'verl'`) |
| conda | `/home/i-fengyicheng/miniconda3/bin/conda` (not on PATH) |
| `uv` | `/usr/local/bin/uv` |
| cores / RAM | 17 cores / 33 GB |
| `rlaunch` | `/kubebrain/rlaunch` present |
| `brainctl` | `/kubebrain/brainctl` present |

Consequence: any stage that needs vLLM rollout or FSDP/Megatron training must run on an
`rlaunch` GPU worker (`--charged-group=codesign --private-machine=group --positive-tags=h800`).
CPU-only stages (data prep, launcher dry-runs, unit tests, verifier scoring, param merge,
analysis) run directly on this node.

## Repo facts worth remembering

- verl vendored version: `0.7.0.dev` (`training/verl/verl/version/version`).
- MOPD-specific code concentrates in:
  - `training/verl/verl/workers/actor/mt_opd.py` (25 KB) — MT-OPD reward/weighting/metrics kernels
  - `training/verl/verl/trainer/ppo/ray_trainer.py` (~2200+ lines) — driver loop + MT-OPD orchestration
  - `training/verl/verl/trainer/ppo/core_algos.py` — `token_reward_direct`, `token_reward_direct_plus_grpo`
  - `training/verl/verl/utils/reward_score/opd_val_dispatch.py`
  - `training/verl/verl/experimental/reward/reward_loop/opd_validation.py`
- `reward_mode` is the central switch: `opd_kl` (default) | `delta_opd` | `mt_opd`.
- MT-OPD teachers are passed as extra Hydra groups `+mt_reward_model_{i}.*`.

## Known gaps in the release (surfaced, not patched)

`evals/README.md` documents modules and asset dirs that are **absent** from this release:

- `evals/rollout_engine/build_data.py` (documented as `python -m evals.rollout_engine.build_data`)
- `evals/misc/` (documented as `python -m evals.misc.fetch_livecodebench_assets`)
- `evals/rollout_engine/scripts/` (documented as `evals/rollout_engine/scripts/run_all.sh`)
- `evals/verifier/third_party/` (`repos.lock.json`, `patches/`)
- `evals/data/` (parquet + benchmark_assets)

So the eval path that actually exists end-to-end is:
`vllm_rollout.py` → `score_rollouts.py` / `verifier/score.py` → `aggregate_lcb_scorecard.py`,
with the caller supplying its own input parquet.
