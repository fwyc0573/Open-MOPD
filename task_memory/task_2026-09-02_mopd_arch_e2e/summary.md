# Task Summary

## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-11 | Hardened reproducibility docs around CPU-built persistent venv, preflight, and one-command H800 reruns. |
| 2026-09-03 | Archived the architecture deliverables, corrected E2E implementation, and final validation evidence. |
| 2026-09-04 | Archived double-check results and scoped repairs for launcher, MT-OPD, GRM, and capability follow-ups. |
| 2026-09-11 | Archived the successful personal H800 E2E using a persistent worker-portable Python 3.12 environment. |

## Task Overview

This task deconstructed the Open-MOPD architecture and exercised its complete local
training/evaluation path on one H800. The run used a fully synthesized four-model Qwen3
family and offline Hugging Face settings, so no released checkpoint or network weight
download was involved.

## Deliverables Inventory

* `/data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e/docs/01_architecture.md`
* `/data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e/docs/02_mt_opd_algorithm.md`
* `/data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e/docs/03_reproduce_e2e.md`
* `/data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e/test_report_2026-09-02_e2e.md`
* `/data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e/e2e/`
* `/data/ycfeng/Open-MOPD/tests/unit/` (task regression tests)
* `/data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e/design.md`
* `/data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e/harness.md`
* `/data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e/review.md`
* `/data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e/lessons.md`
* `/data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e/future.md`
* `/data/ycfeng/Open-MOPD/experiments/tests/test_if_rl_grm.py`
* `/data/ycfeng/Open-MOPD/experiments/analysis/capability_subspace.py`
* `/data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e/test_report_2026-09-11_personal_h800_persistent_venv.md`

## Validation Status

| Gate | Outcome | Metrics |
| ---- | ------- | ------- |
| Open-MOPD tests | PASS | `132 passed` |
| Focused code regressions | PASS | `5 passed` (attention/entropy) |
| Future-work MT-OPD regressions | PASS | `60 passed` |
| Future-work GRM regressions | PASS | `17 passed` |
| Future-work capability regressions | PASS | `69 passed` |
| Config probe | PASS | `0` unexpected outcomes |
| SFT | PASS | 6 steps; train loss `6.4359403 -> 5.8885908`; val loss `6.3142653 -> 5.7854004` |
| Merge | PASS | `model.safetensors` written, `885032` bytes |
| RL | PASS | 2 steps; throughput `397.2633` and `695.6351` tokens/s |
| OPD | PASS corrected path | corrected exit `0`; reward means `-0.5174221`, `-0.4962008` |
| MT-OPD | PASS corrected paths | 3 domains; M1 weights code/if/math `1.3697561/0.5695741/1.1209581` |
| Eval | PASS | 12 rows, 64 completion tokens, verifier exit `0` |
| Optional repo suite | KNOWN FAILURES | `111 passed, 8 failed`; GRM fixtures and one tolerance test |
| Personal H800 E2E (`exp-0911-152629-174180`) | PASS | Worker and launcher terminal status succeeded; all stages exit 0; 12-row offline vLLM rollout |

## Open Items/Future Extensions

The local OPD/MT-OPD launchers now contain the Hydra `+` and `log_prob_top_k` fixes, and
MT-OPD validates positional teacher/domain binding before routing. M2's `divide` default
remains unchanged because switching to the measured-preferred `multiply` changes training
semantics and needs an explicit migration decision. The verifier still has no `dummy_math`
scorer, so the offline eval records artifact/schema evidence only. The unrelated repo-suite
failures remain documented in the original test report.


The persistent environment uses `torch 2.8.0+cu128`, `vllm 0.11.0`, `transformers 4.57.6`, `ray 2.55.1`, editable `verl`, IF dependencies, and locally staged NLTK tokenizers. The verifier result remains `unknown_dataset` for synthetic `dummy_math`; this E2E is a control path and dependency validation, not a benchmark quality or convergence result.

The repeat-run contract is documented in `docs/03_reproduce_e2e.md`: repair and preflight
the worker-portable venv on the CPU host, then submit one local StepMind RJob using the
mounted environment. `task_memory/env_handbook.md`, `harness.md`, and `notes.md` record
the same rule so future runs do not install dependencies dynamically on GPU workers.
