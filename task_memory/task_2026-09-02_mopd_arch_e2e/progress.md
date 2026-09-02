# Progress Log

## Modification History

| Date       | Summary of Changes                                     |
| ---------- | ------------------------------------------------------ |
| 2026-09-02 | Session 1: scouting, architecture analysis, E2E scripts, config probe |
| 2026-09-03 | Final r5 H800 E2E completed; regression fixes and required reports/guides added |

## Status board

| # | Item | Status |
| --- | --- | --- |
| S0 | scout repo + host env | **completed** |
| S1 | architecture deep-read workflow (12 subsystems, 29 agents) | **completed** |
| S1b | independent re-vet workflow (48 claims, 10 agents) | **completed** |
| S2 | dummy-asset generator + stage scripts | **completed** |
| S2b | Hydra compose-only probe (`15_config_probe.py`) | **completed — 11/11 as expected** |
| S6 | `docs/01_architecture.md`, `docs/02_mt_opd_algorithm.md` | **completed** |
| S3 | NFS venv install (`00_setup_env.sh`) | **completed** — H800 worker imports verl/vllm/ray |
| S4 | GPU stage runs on an rlaunch H800 | **completed** — r5 core stages passed |
| S5 | test report | **completed** — numeric r5 evidence and known failures recorded |
| S7 | `docs/03_reproduce_e2e.md` | **completed** — copyable commands and artifacts documented |
| S8 | task archive docs | **completed** — design/harness/review/lessons/future/summary |

---

## 2026-09-02 — findings, in the order they were established

### F1. Host has no GPU; `rlaunch` is the only path to one

`nvidia-smi` absent, `torch.cuda.is_available() == False`, no `transformers`/`vllm`/`ray`,
`verl` not importable. `/kubebrain/rlaunch` present. User chose the GPU-first
high-fidelity path (see `requirements.md` Q1 resolution).

### F2. `/data` is Ceph RBD, and parallel agent fan-out starves the installer

`df -hT` → `/data` on `/dev/rbd1` (xfs), 88% full. During the two verification workflows
the host showed `%Cpu(s): 90.3 wa` and load average **235**, and `uv`'s unpack rate fell to
**32 MB/min**. My own `du -sb` polling on a ~300k-file cache tree was part of the problem.

**Motivation** for the change: finish the install this session.
**Expectation:** poll only the log file; unpack rate recovers once the workflows drain.
**Method:** replaced `du`-based polling with `grep` on the log; serialized the remaining work.
**Result:** rate recovered; install progressed from 4 to 9 completed large wheels.

### F3. Two shipped launchers cannot run — Hydra rejects them **[executed]**

`scripts/local/opd.sh:66` and `scripts/local/mt_opd.sh:87` emit
`actor_rollout_ref.rollout.reward_mode=<mode>` with **no leading `+`**, while the same
`mt_opd.sh` correctly uses `+` for every `mt_opd.*` key. `reward_mode` is declared in no
YAML under `verl/trainer/config/` and is absent from the `RolloutConfig` dataclass.

Compose-only probe output (re-run independently):

```
[baseline] COMPOSED OK
  rollout has reward_mode key: False
  rollout has log_prob_top_k key: False
  rollout.get('log_prob_top_k', 0) -> 0
[plain reward_mode (as shipped)] RAISED ConfigCompositionException: Could not override
    'actor_rollout_ref.rollout.reward_mode'.
    To append to your config use +actor_rollout_ref.rollout.reward_mode=mt_opd
[+reward_mode] COMPOSED OK
  reward_mode value: mt_opd
[+mt_opd.teacher_domains] COMPOSED OK
```

This fires at composition time — before Ray starts, before any GPU is touched.

### F4. `log_prob_top_k` reads 0 and MT-OPD raises on it **[executed]**

Same probe: the key is absent from the composed config, so `ray_trainer.py:1968`'s
`.get("log_prob_top_k", 0)` returns 0 and `:1973-1974` raises for `mt_opd`/`delta_opd`. The
`256` at `workers/config/rollout.py:154` is the dataclass default and reaches only the vLLM
engine. No launcher sets it.

### F5. SFT parquet needs a `messages` column, not `question`/`answer`

**Motivation:** my first `30_sft.sh` passed `data.prompt_key=question data.response_key=answer`.
**Root cause:** `verl/trainer/sft_trainer.py:378` is
`@hydra.main(config_name="sft_trainer_engine")`, so `sft_trainer.yaml` (which declares
`prompt_key`/`response_key`) belongs to the *other* trainer, `fsdp_sft_trainer.py`.
`sft_trainer.py:393` always builds `MultiTurnSFTDataset`, which reads its column name from
`config["multiturn"]["messages_key"]` (`multiturn_sft_dataset.py:62-66`) while
`sft_trainer_engine.yaml` declares `messages_key` flat under `data:` (`:24-27`) — a block the
nested lookup never sees. Effective column name: the hard default `"messages"`.
**Method:** rewrote the SFT parquet in `10_make_dummy_assets.py` to emit
`messages/domain/source/dataset/split/difficulty/sample_id`, mirroring the released MixSFT
schema; removed the two dead overrides.
**Result:** probe case "sft (as scripts/local/sft.sh emits it)" → COMPOSED OK; the
`prompt_key` variant → RAISE, as expected.

### F6. Stage scripts restructured to drive the released launchers

**Motivation:** the user must be able to reproduce the *released* entrypoints, and the
launcher defects should surface rather than be papered over.
**Method:** every stage now calls `scripts/local/<stage>.sh --run` with `--extra` overrides.
For OPD and MT-OPD, which cannot run that way (F3), each script has two labelled parts:
`50a`/`60a` run the shipped launcher and record the exact `ConfigCompositionException`;
`50b`/`60b` run a corrected direct `main_ppo` invocation so the algorithm path is exercised.
**Result:** 11/11 probe cases land on their expected verdict, 0 unexpected.

### F7. Config probe added as a mandatory pre-GPU gate

`15_config_probe.py` composes every stage's exact override list with `hydra.compose()` and
no execution. A wrong key now costs a second instead of ten minutes of H800 time (Ray boot
+ vLLM KV allocation + four FSDP model loads). Results:

| Case | Expected | Actual |
| --- | --- | --- |
| ppo baseline | COMPOSE | COMPOSE |
| sft (as `scripts/local/sft.sh` emits it) | COMPOSE | COMPOSE |
| sft with `prompt_key`/`response_key` | RAISE | RAISE |
| rl grpo (as `scripts/local/rl.sh` emits it) | COMPOSE | COMPOSE |
| opd AS SHIPPED (`opd.sh:66`) | RAISE | RAISE |
| opd CORRECTED (`+reward_mode`, `+log_prob_top_k`) | COMPOSE | COMPOSE |
| mt_opd AS SHIPPED (`mt_opd.sh:87`) | RAISE | RAISE |
| mt_opd CORRECTED (naive M-OPD) | COMPOSE | COMPOSE |
| mt_opd CORRECTED + M1 | COMPOSE | COMPOSE |
| mt_opd CORRECTED + M1 + M2 + M3 | COMPOSE | COMPOSE |
| `+mt_opd.target_gradient_shares={...}` dict literal | unknown | **COMPOSE** |

The last row settles an open question: the source comment at `ray_trainer.py:671-673` says
a dict-valued override for `target_gradient_shares` "does not compose on the executor". It
composes fine in Hydra — so the failure was shell quoting on the way in, not Hydra.

### F8. Architecture analysis: 43/48 load-bearing claims confirmed

The 12-subsystem deep read was re-vetted claim-by-claim by 9 independent skeptics plus an
adjudicator: **43 CONFIRMED, 4 IMPRECISE, 1 WRONG**. I then re-read the two highest-stakes
files myself end-to-end (`mt_opd.py` all 563 lines, `ray_trainer.py:2096-2235`) and every
claim about them held exactly, including line numbers.

The one WRONG: `build_mix_rl_raw_union.py` is *not* the only in-repo builder emitting a
top-level `domain` column (`build_mix_rl_union.py:108` and
`training/scripts/sft/build_weighted_mix.py:15` also do). The part that matters survives:
no builder under `training/scripts/rl/` emits it, so an RL-only parquet fails MT-OPD.

Full ledger: `raw/corrections_ledger.md`. Raw subsystem reports: `raw/subsystem_reports.json`.

---

## 2026-09-03 — final r5 verification

### F9. Final worker result

`openmopd-e2e-full-20260903-r5` ran on an H800 with the offline dummy assets. The
Open-MOPD suite passed `132` tests. SFT, merge, RL, corrected OPD, corrected MT-OPD
(`naive` and `m1`), and eval all completed with exit `0`. The wrapper exit was `1`
because the optional repo suite reported `111 passed, 8 failed`; the failures are
outside the MOPD core and remain visible in the report.

### F10. Central MT-OPD evidence

At M1 step 2, prompt shares were `1/3` for each domain while token shares were code
`0.2920227945`, IF `0.3511396050`, and math `0.3568376005`. The corresponding loss
weights were code `1.3697561026`, IF `0.5695740581`, and math `1.1209580898`.
Teacher conflict metrics were finite (`spread_mean=0.3826360404`,
`contested_frac=0.0028490028`).

### F11. Documentation and regression coverage

Added `docs/03_reproduce_e2e.md`, `test_report_2026-09-02_e2e.md`, `design.md`,
`harness.md`, `review.md`, `lessons.md`, `future.md`, and `summary.md`. Focused
regressions for the attention backend and entropy layout report `5 passed` locally.

### Modification record — final documentation and verification

* **Motivation:** close the two user-requested deliverables and preserve an auditable
  record of the final worker run.
* **Expectation:** a reader can reproduce the offline chain from one guide, and every
  acceptance criterion has a numeric result or an explicit blocker.
* **Method:** added the reproduction/test/archive documents, linked each result to the
  r5 log or artifact path, and reran the five focused regression tests.
* **Result:** all required documents exist with modification histories; the focused
  tests exit `0` (`5 passed`); the core E2E stages exit `0`, while the eight unrelated
  repo-test failures remain explicitly reported.

## Artifacts produced so far

```
task_memory/task_2026-09-02_mopd_arch_e2e/
├── requirements.md              user intent, verbatim
├── plan.md                      scope, dependency map, acceptance criteria
├── notes.md                     host env + release gaps
├── progress.md                  this file
├── issues.md                    open items and blockers
├── docs/
│   ├── 01_architecture.md       R1: layers, modules, call chain, data contracts
│   └── 02_mt_opd_algorithm.md   R1: the algorithm, config surface, traps
├── e2e/
│   ├── 00_setup_env.sh          NFS venv build
│   ├── 10_make_dummy_assets.py  tokenizer + 4 tiny models + all parquet
│   ├── 15_config_probe.py       compose-only Hydra gate
│   ├── 20_unit_tests.sh         the Open-MOPD test suite
│   ├── 25_dry_runs.sh           all 5 launchers, argv + negative checks
│   ├── 30_sft.sh                MixSFT
│   ├── 35_merge.sh              FSDP shards -> loadable HF dir
│   ├── 40_rl.sh                 domain RL (GRPO)
│   ├── 50_opd.sh                single-teacher OPD (shipped + corrected)
│   ├── 60_mt_opd.sh             MT-OPD (shipped + corrected; naive|m1|full)
│   ├── 70_eval.sh               vLLM rollout + scoring
│   ├── _common.sh               shared settings
│   └── run_all.sh               ordered driver
└── raw/
    ├── architecture_map.md      the synthesized map
    ├── completeness_critique.md
    ├── corrections_ledger.md    the re-vet ledger
    ├── revet_verdicts.json      48 verdicts
    ├── subsystem_reports.json   12 structured reports + verifier verdicts
    └── trace_*.md               3 end-to-end traces
```
