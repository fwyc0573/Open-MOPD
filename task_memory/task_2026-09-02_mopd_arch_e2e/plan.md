# Execution Plan

## Modification History

| Date       | Summary of Changes                                                       |
| ---------- | ------------------------------------------------------------------------ |
| 2026-09-10 | Record personal Python H800 reproduction and verified runtime observations. |
| 2026-09-02 | Initial plan; scope fixed by user decision (GPU-first high-fidelity E2E) |
| 2026-09-04 | Added continuation plan for evidence-led future-work double-check and scoped repairs. |

## Scope (fixed by user decision 2026-09-02)

- **R1 docs**: architecture deconstruction of the core MOPD codebase, landed as docs.
- **R2 E2E**: high-fidelity single chain on an `rlaunch` H800 worker, using a
  **fully locally synthesized** dummy model family (no HF cache, no weight download).
- Deliverable for R2 is a **manually reproducible guidance doc** with every shell command.

## Subtask dependency map

```
S0 (scout repo + env)  -- DONE
   |
   +-> S1 (architecture deep-read workflow)  ──┐   runs in parallel with S2..S4
   |                                            |
   +-> S2 (write asset generator + run scripts) |
         |                                      |
         v                                      |
       S3 (rlaunch worker + env install)         |
         |                                       |
         v                                       |
       S4 (E2E stage runs: unit tests -> SFT -> RL -> OPD -> MT-OPD -> eval)
         |                                       |
         v                                       v
       S5 (test report) <──────────── S6 (architecture docs)
         |                                       |
         +──────────────> S7 (reproduction guidance doc) <─┘
```

Arrow chain: `S0 -> {S1, S2} ; S2 -> S3 -> S4 -> S5 ; S1 -> S6 ; {S5, S6} -> S7`
S1 runs in parallel with S2/S3/S4.

## Continuation — future-work double-check (2026-09-04)

Dependency map: `F0 (read task records) -> {F1 launcher repair, F2 teacher/binding repair, F3 GRM/capability repair} -> F4 conflict wiring and M2 decision -> F5 docs/tests/archive`.

`F1`, `F2`, and `F3` are independent source areas and may run in parallel after `F0`;
`F4` consumes the teacher/metric evidence; `F5` waits for every verification result.

Acceptance: every future item has a verdict tied to a file/runtime observation; fixes stay
within local launcher/test/trainer boundaries; behavior-changing defaults and production
scorer additions remain explicitly deferred when evidence does not settle the decision.

## S1 — Architecture deep-read (in flight, workflow `wf_4fa2f58c-0e7`)

12 parallel subsystem readers, each adversarially verified, plus 3 end-to-end
traces, plus a synthesis and a completeness critic. Subsystems:
`mt_opd_kernels`, `ray_trainer_driver`, `core_algos_adv`, `rollout_reward_mode`,
`teacher_workers`, `config_schema`, `sft_stage`, `rl_data_prep`, `evals_layer`,
`experiments_layer`, `tests_as_spec`, `launchers_and_deps`.

Acceptance: every claim in the final map carries a `file:line`; the completeness
critic finds no missing core mechanism.

## S2 — Dummy assets + run scripts

All scripts live under `task_memory/task_2026-09-02_mopd_arch_e2e/e2e/` so they
survive a GPU-worker reclaim (NFS-backed).

| Script | Purpose |
| --- | --- |
| `00_setup_env.sh` | build NFS venv `/data/ycfeng/envs/openmopd-py312`, install verl + deps |
| `10_make_dummy_assets.py` | synthesize tokenizer + 4 tiny Qwen3 models + all parquet inputs |
| `20_unit_tests.sh` | pytest the Open-MOPD test set (the executable spec) |
| `25_dry_runs.sh` | all 5 launchers in dry-run mode (argv construction check) |
| `30_sft.sh` | MixSFT stage on the dummy SFT parquet |
| `40_rl.sh` | GRPO domain-RL stage |
| `50_opd.sh` | single-teacher OPD (`reward_mode=opd_kl`) |
| `60_mt_opd.sh` | multi-teacher OPD (`reward_mode=mt_opd`, 3 teachers, 3 domains) |
| `70_eval.sh` | vLLM eval rollout + `score_rollouts` + verifier |
| `run_all.sh` | ordered driver |

Dummy model family (4 copies, each with a different random seed so teacher
disagreement metrics are non-degenerate):

```
Qwen3ForCausalLM  hidden_size=64  num_hidden_layers=2
                  num_attention_heads=4  num_key_value_heads=2  head_dim=16
                  intermediate_size=128  vocab_size=1024
                  max_position_embeddings=4096
tokenizer: ByteLevel BPE synthesized on the spot (byte alphabet + merges)
           + explicit chat_template
```

## S3 — GPU worker

```bash
rlaunch --predict-only --charged-group=codesign --private-machine=group \
        --positive-tags=h800 --gpu=1 --cpu=16 --memory=131072 -- bash -lc 'true'
rlaunch --charged-group=codesign --private-machine=group --positive-tags=h800 \
        --gpu=1 --cpu=16 --memory=131072 --backoff-limit=1 -- bash
```

Acceptance: `nvidia-smi -L` shows an H800; the NFS venv imports `verl`, `vllm`, `ray`.

## S4 — Stage runs

Every stage must produce observable evidence, not just exit 0:

| Stage | Evidence required |
| --- | --- |
| unit tests | pytest PASS count per file, no skips silently swallowed |
| dry-runs | printed argv for each of the 5 launchers |
| SFT | loss values across steps + checkpoint dir listing |
| RL | `critic/score/mean`, `actor/pg_loss`, step timing |
| OPD | `reward_mode=opd_kl` accepted; per-token reward stats |
| MT-OPD | `mt_opd/domain/<d>/prompt_share`, `/token_share`, `/loss_weight` for all 3 domains; teacher-conflict metrics; checkpoint written |
| eval | rollout parquet with `completion` column; `scores.json` |

## S5/S6/S7 — Docs

| File | Content |
| --- | --- |
| `docs/01_architecture.md` | R1: layer map, module catalog, call chain, I/O contracts |
| `docs/02_mt_opd_algorithm.md` | R1: the MT-OPD step in depth + config surface |
| `docs/03_reproduce_e2e.md` | R2: the manual reproduction guide, command by command |
| `test_report_2026-09-02_e2e.md` | R2: results + evidence with actual numbers |

## Acceptance criteria (whole task)

1. Every architecture claim cites `file:line`.
2. The MT-OPD path runs with real teacher routing and emits per-domain metrics
   with concrete numeric values recorded.
3. A reader can copy the commands from `docs/03_reproduce_e2e.md` verbatim and
   land in the same place, with zero weight downloads.
4. Any stage that cannot run is reported as a blocker with its root cause — not
   worked around silently.

## 2026-09-10 reproduction continuation

Dependency: runtime verification -> image launcher -> full offline chain -> artifact audit -> report and guide. Preserve the historical SFT, merge, RL, OPD, MT-OPD naive/M1 and evaluation scope; do not substitute a platform probe for E2E completion. Use the image runtime only after verifying imports and executable paths. Record failed stages and correct their demonstrated causes.
