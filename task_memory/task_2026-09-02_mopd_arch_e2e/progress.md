# Progress Log

## Modification History

| Date       | Summary of Changes                                     |
| ---------- | ------------------------------------------------------ |
| 2026-09-11 | Promote the persistent CPU-built venv and one-command H800 rerun to the documented default. |
| 2026-09-10 | Record personal Python H800 reproduction and verified runtime observations. |
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

## 2026-09-04 — future-work double-check and scoped repairs

### F12. Launcher, teacher binding, and conflict threshold

* **Motivation:** shipped OPD/MT-OPD launchers used undeclared Hydra keys without `+`,
  omitted `log_prob_top_k`, and left teacher-0 settings inconsistent with additional
  teachers; the trainer also failed to enforce positional domain contracts and dropped
  `conflict_nats` at metric call time.
* **Expectation:** current launchers compose, teacher configuration is symmetric, invalid
  positional/domain inputs fail before routing, and policy/metric thresholds agree.
* **Method:** appended runtime keys in both launchers; aligned teacher-0 tokenizer,
  remove-padding, and FSDP offload settings; added `_validate_mt_teacher_binding` and
  unknown-domain rejection in `ray_trainer.py`; passed `self.mt_conflict_nats` to
  `compute_teacher_conflict_metrics`.
* **Result:** launcher dry-run contains `+reward_mode` and `+log_prob_top_k=256`; Hydra
  probe reports `unexpected outcomes: 0`; MT-OPD tests pass `60`; invalid plain reward
  mode remains a deliberate negative regression.

### F13. GRM and capability-subspace test correctness

* **Motivation:** GRM tests either supplied a FakeSession without `.mount()` or patched a
  helper bypassed by concurrent fan-out, causing DNS retries and false failures; explicit
  capability directions used randomized SVD against an exact-SVD reference.
* **Expectation:** tests isolate the actual transport seam and explicit directions remain
  deterministic across calls.
* **Method:** added FakeSession `.mount()`, stubbed `_request_completion` in semantic tests,
  and switched small explicit direction construction to deterministic `torch.linalg.svd`.
* **Result:** `test_if_rl_grm.py` passes `17`; `test_capability_subspace.py` passes `69`.

### F14. Deferred decisions

* **Motivation:** M2 `reward_scale_direction` has a real outcome-changing default choice;
  `dummy_math` lacks a production scorer by design.
* **Expectation:** preserve compatibility while recording evidence-based recommendations.
* **Method:** kept default `divide`, documented explicit `multiply` recommendation and the
  need for a migration decision; left scorer registry unchanged and retained the existing
  `unknown_dataset` offline evidence.
* **Result:** no unapproved cross-cutting behavior change; both items are recorded in
  `future.md` as deferred.

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

## 2026-09-10 — personal StepMind RJobBackend H800 rerun

- Loaded `/data/ycfeng/codex-home/skills/stepmind-gpu-worker/SKILL.md` and the mandatory StepMind Python RJobBackend runbook.
- Configured personal `i-fengyicheng` credentials in a fresh Python process with `STEPMIND_BACKEND=rjob`; no local `rlaunch`/`brainctl` submission was used after the new runbook took effect.
- H800 1-GPU mount probe passed with RJob `exp-0910-184007-973388` and terminal status `succeeded`.
- Full rerun attempts:
  - `exp-0910-184114-313429`: failed because source mount was `/data/ycfeng/steptron` while target was `/data/ycfeng/Open-MOPD`.
  - `exp-0910-184251-104295`: source/target `/data/ycfeng/Open-MOPD`, but command quoting produced `unexpected EOF`.
  - `exp-0910-184442-331302`: quoting fixed; source/target remained too narrow for the external venv/assets and failed before useful logs.
  - `exp-0910-184734-511737`: attempted parent mount but helper semantics used source `/data/ycfeng/Open-MOPD` -> target `/data/ycfeng`; failed with no business log.
  - `exp-0910-184920-144962`: submission cwd `/data/ycfeng`, annotation verified as `100.96.128.189:/data/ycfeng:/data/ycfeng`, creator `i-fengyicheng`, H800 worker allocated; task still failed during startup with zero log rows. Platform status was `RJobFailed` and no command-level traceback was exposed.
- The historical r5 artifacts are absent; therefore the 2026-09-10 personal-backend attempt is not claimed as a fresh complete E2E pass. Historical r5 numeric evidence remains in `test_report_2026-09-02_e2e.md`.

### 22:48 image runtime retry

Probe exp-0910-222602-152735 succeeded on H800 with /usr/bin/python, torch 2.8.0+cu128 and vllm 0.11.0.post37. The old setup failure is confirmed as uv missing (exit 127); previous guesses that zero API log rows implied mount failure were unsupported. The remote read-only log path exposed the actual error. Direct rjob_run previously omitted mount annotation and pull_code=False; new submissions return to the required spawn_tasks path.

Created tests/e2e/submit_h800_image.py and tests/e2e/run_h800_image.sh. Added an executable-path override for TORCHRUN in e2e/_common.sh, corrected the command to generator --out, and retained the existing stage scripts. Syntax checks pass. Current job exp-0910-224825-381481, local launcher session 36114, output /data/ycfeng/tmp/openmopd-image-20260910-224825. Status: in-progress; no E2E pass claimed.

At 2026-09-10 14:53:45 UTC, exact-job API confirms exp-0910-224825-381481 remains Pending / RJob is queuing with no replica. Launcher session 36114 is live. This continuation is a verified wait after concrete launcher corrections and submission, not an E2E pass. Next action: poll the same job and inspect WORK/logs/worker.log when started. No permission request or additional submission is needed.

Commit attempt could not proceed: existing .git/index.lock (zero bytes, mtime 2026-09-07) rejected git add and git commit. A git status --porcelain process was visible; fuser is unavailable, so lock ownership remains unproven. The lock was preserved and no process was terminated. Code changes remain uncommitted; this does not stop the existing RJob. Resume with exact-job inspection before any new submission.

At 2026-09-10 15:08:08 UTC, exact-job observation confirms `exp-0910-224825-381481` remains Pending. Queue metadata is authoritative: `queue-name=codesign-default`, `queue-status=suspend`, ranking `3`, reason `Insufficient GPU quota. Request: task default 1 GPU/replica x 1 replicas = 1 GPU (H800). Queue remaining: H800=0.` Launcher session 36114 remains alive. No duplicate submission was made. This is an external quota wait, not a workload failure.

### 2026-09-10 15:27:51 UTC — verified queue wait and guide update

Exact-job Python API still reports exp-0910-224825-381481 Pending. Queue annotation read in this continuation reports codesign-default, ranking 3, H800 remaining 0. Local Python launcher PID 1341110/session 36114 remains alive. No replica/business output and no duplicate submission. Updated docs/03_reproduce_e2e.md with the current personal spawn_tasks command, exact-job inspection, local mount and lifetime rules, current unverified E2E status, and the supported --out argument. Marked the old rlaunch sections as historical reference only. git diff --check passes. The pre-existing Git lock remains unresolved; no lock or process was removed.

## 2026-09-11 — temporary H200 route

User directed a temporary H200 run while H800 is unavailable. Read `stepmind-python-rjob.md`, the verified H200 quota task record, and `vllm-bs-0.10.2-frontier-env.md`. Confirmed differences: H200 uses `infer_af_test` or task-specific `step_main`, `H200` tag, valid RJob names use hyphens, and the pinned H200 image may have a zero-byte `/usr/bin/nvidia-smi`; use `/usr/local/nvidia/bin/nvidia-smi`.

A personal Python probe using `infer_af_test` was rejected by the platform: that group is currently locked. A second probe using task-specific `step_main`, `H200`, and image `hub.i.basemind.com/vllm-0.10.2/frontier-env@sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc` was accepted as RJob `exp-0911-011401-249803`. API confirms creator `i-fengyicheng`, queue `step-main-default`, scheduled and pod-created annotations, and status `Starting`; replica logs are not available yet. It is a runtime/image probe, not the complete MOPD E2E.

## 2026-09-11 H200 continuation

- `exp-0911-012823-968281` (H200, `step_main`) reached worker but failed unit/config preflight because `tensordict`, `omegaconf`, and `hydra` were absent.
- Dependency iteration jobs: `exp-0911-013313-922823` (incorrectly resolved Torch 2.14 dependency), `exp-0911-013856-585464` (ANTLR 4.13/OmegaConf mismatch), and `exp-0911-014002-121513` (`tensordict` missing `pyvers`).
- Current job: `exp-0911-014420-508385` on H200 with `tensordict==0.10.0`, `hydra-core`, `omegaconf`, `antlr4-python3-runtime==4.9.3`, `pyvers`, `cloudpickle`, `torchdata==0.11.0`, and explicit `/usr/local/bin/torchrun`. GPU, dummy assets, unit tests, dry-runs, and config probe passed; SFT launched. Later RL/OPD stages currently expose another image dependency gap: `ModuleNotFoundError: codetiming`.

### E2E standardness audit (2026-09-11)

- Standard execution paths: SFT/RL/OPD/MT-OPD invoke the repository launchers and `verl` trainers; evaluation invokes the repository vLLM rollout and scoring modules.
- Synthetic scope: `10_make_dummy_assets.py` creates four ~0.44M-parameter seed-distinct Qwen3 models and synthetic math/code/IF parquet rows. This preserves schema, teacher disagreement, routing, loss, and gradient code paths, but is not a real benchmark dataset or released model.
- Rollout status: vLLM rollout path is technically active and previously emitted rollout parquet. Current inputs are synthetic `eval_input.parquet`; IF/LCB real verifier metadata is intentionally absent and dispatch falls back to math scoring. Therefore rollout correctness is interface/path evidence only, not real benchmark quality evidence.
- Scale reductions: single H200 GPU, TP/DP=1, eager attention, no-remove-padding, short prompt/response limits, tiny batch/micro-batch, and a few training steps. These are recorded as smoke/runtime settings; they preserve high-level loss/gradient/routing semantics but do not support convergence, throughput, or production loss claims.

### 2026-09-11 dependency iteration observation: exp-0911-014911-871996

Exact-job personal Python API inspection reports `phase=Stopped`, replica `exp-0911-014911-871996-cfcd2084` in `FAILED`. The worker successfully installed the requested non-Torch runtime libraries (including `codetiming`, `datasets`, `hydra-core`, `omegaconf`, `tensordict`, `ray`, and related packages), then failed during `10_make_dummy_assets.py` before business stages. The first traceback is:

```text
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
```

The importer is `transformers 5.17.0`, while the image provides an older incompatible `huggingface_hub` under `/usr/local/lib/python3.10/dist-packages`. Repository setup pins `transformers==4.57.6` (`training/install_requirements.sh:17`, `e2e/00_setup_env.sh:41`), so the next dependency iteration must pin that version (or an explicitly compatible 4.x release) and verify import before rerunning. The failed job log snapshot is preserved at `/data/ycfeng/tmp/exp-0911-014911-871996.log`; no remote submission or control was performed during this observation.

### 2026-09-11 latest min-libs run: exp-0911-121153-268088

The next H200 iteration retained the image's `transformers 4.57.3` and installed only the minimal runtime set. It reached the vLLM evaluation path: vLLM 0.11.0.post37 loaded the local dummy model, processed 12 prompts, and wrote a 12-row rollout parquet with `finish_reason=length`. However, unit collection and all training stages failed on `ModuleNotFoundError: peft`; the minimal dependency list omitted `peft` (and its `safetensors` dependency). The repo-level tests also hit an independent offline-environment issue: instruction-following tests attempted NLTK corpus download through an unauthenticated proxy and failed with HTTP 407. The log ends at the `70b — score_rollouts (CPU)` header after the successful 70a rollout, so no scoring PASS is claimed. Exact-job API state is `phase=Failed`, replica `FAILED`; full log is `/data/ycfeng/tmp/exp-0911-121153-268088.log`.

### 2026-09-11 active min-libs+peft run: exp-0911-121620-993332

Exact-job API reports `phase=Running`, replica `exp-0911-121620-993332-cfcd2084` `RUNNING`. H200 worker setup installed `peft==0.15.2` but did not include its transitive `accelerate`; unit collection and SFT/RL/OPD/MT-OPD fail at PEFT import with `ModuleNotFoundError: accelerate` (`peft/utils/loftq_utils.py` imports `accelerate.utils.memory.clear_device_cache`). The repository's `training/requirements-if-rl.txt` also requires `nltk>=3.8`, `langdetect>=1.0.9`, `absl-py>=1.0.0`, and `immutabledict>=2.0.0`; these are needed for the instruction-following reward tests and should be installed in the next iteration. The 70a vLLM rollout still starts independently; status is not yet terminal while this observation was captured.

- `exp-0911-121153-268088`: minimal dependency run reached RL/OPD/MT-OPD; failed at `peft` import because `accelerate` was absent. vLLM eval loaded local dummy model with `HF_HUB_OFFLINE=1`.
- `exp-0911-121620-993332`: added `peft==0.15.2`; failed because `accelerate` was absent inside peft.
- `exp-0911-121950-088240`: added accelerate; platform stopped the worker before a durable worker log was emitted, so no business-level diagnosis is available.
- Standardness audit saved as `standardness_audit_2026-09-11.md`; do not claim standard training progression until checkpoint chaining, real reward metadata, and verifier gating are corrected.

Follow-up status for `exp-0911-121620-993332`: the job is now `phase=Stopped`, replica `STOPPED`, with worker `ExitCode=-1` after 70a. `peft==0.15.2` installation succeeded, but `accelerate` was omitted; every training entry point fails at `peft/utils/loftq_utils.py` importing `accelerate.utils.memory`. The 70a vLLM rollout completed 12 prompts and emitted the expected parquet; the process reached score stage without a scoring completion marker. Repo-level IF reward tests continue to fail because `_ensure_nltk_data()` attempts direct GitHub download and receives proxy HTTP 407. Next iteration should install `accelerate` plus `nltk`, `langdetect`, `absl-py`, and `immutabledict`, and pre-stage `punkt_tab` locally so scoring stays offline.

## 2026-09-11 switch back to H800

- H200 `exp-0911-122529-289953` reached vLLM rollout successfully (12 rows, local dummy model, offline mode) but failed later in the dependency/training chain; H200 route is now disabled per user instruction.
- Existing H800 fallback `exp-0910-224825-381481` is stopped and was not reused.
- New H800 job submitted via local StepMind Python RJobBackend: `exp-0911-123411-341638`, route `codesign + H800`, local NFS `100.96.128.189:/data/ycfeng:/data/ycfeng`, personal creator `i-fengyicheng`.

## 2026-09-11 wrapper persistent-venv update

- Updated `tests/e2e/run_h800_image.sh` to consume the CPU-host-built `/data/ycfeng/envs/openmopd-py312` and its `torchrun`; the worker now fails fast when either executable is absent and performs all probes/asset generation with the venv interpreter. Runtime dependency installation on the ephemeral worker is no longer part of this wrapper.
- Added a preflight import probe for `vllm`, `transformers`, `ray`, `tensordict`, `hydra`, `verl`, and `verifiable_instructions`.
- Verification: `bash -n tests/e2e/run_h800_image.sh` PASS. Local venv imports PASS: torch `2.8.0+cu128`, vLLM `0.11.0`, transformers `4.57.6`, ray `2.55.1`, tensordict `0.10.0`, hydra `1.3.6`, verl `0.7.0.dev`, verifiable_instructions `0.1.0`; NLTK `punkt` and `punkt_tab` are present. Local host reports `cuda_available=False`, expected because this check runs on CPU master.

## 2026-09-11 — H800 persistent-venv E2E PASS

- Built `/data/ycfeng/envs/openmopd-py312` on the CPU host using the mirror-backed `e2e/00_setup_env.sh`. Verified `torch 2.8.0+cu128`, `vllm 0.11.0`, `transformers 4.57.6`, `ray 2.55.1`, `tensordict 0.10.0`, `hydra 1.3.6`, `verl 0.7.0.dev`, `verifiable_instructions 0.1.0`, `pylatexenc`, `accelerate`, and IF dependencies. Prepared NLTK `punkt` and `punkt_tab` under the persistent venv.
- Fixed worker portability: uv's default `bin/python` symlink targeted `/home/brainpp/.local` and was invisible in GPU workers. Copied the CPython runtime into `/data/ycfeng/envs/cpython-3.12.13-linux-x86_64-gnu` and repointed the venv launcher. `00_setup_env.sh` now performs this conversion automatically; `tests/e2e/run_h800_image.sh` uses the persistent `PY`/`TORCHRUN` and fails clearly if absent.
- Submitted locally through personal StepMind Python `RJobBackend` (`STEPMIND_BACKEND=rjob`, creator `i-fengyicheng`), `codesign + H800`, 1 GPU, 16 CPU, 132 GB, explicit megatron-step image, local NFS `100.96.128.191:/data/ycfeng:/data/ycfeng`.
- RJob `exp-0911-152629-174180` reached `Succeeded`; launcher reported `FINAL_STATUS succeeded`, exit marker `0`. Worker verified H800 allocation and persistent Python runtime.
- Stage evidence: unit tests PASS (133 tests plus 120 repo/eval tests), dry-runs PASS, config probe PASS, SFT PASS, merge PASS, RL PASS with loss/gradient/throughput metrics, OPD PASS, MT-OPD shipped and corrected M1 PASS, vLLM rollout PASS (12 parquet rows), score command PASS, verifier command PASS. Checkpoints and merged model are retained under `/data/ycfeng/tmp/openmopd-image-20260911-152629/runs`.
- Verifier output is explicitly `dummy_math: unknown_dataset, scored_rows=0`; this is an expected synthetic-data limitation and is not a benchmark reward/quality result. Rollout is technically healthy (local dummy model, offline flags, 12 generated rows), while standard dataset semantics remain out of scope because no released weights or external dataset downloads are allowed.
- A transient Ray/multiprocess cleanup `Device or resource busy` traceback appeared in logs, but all corrected MT-OPD training metrics completed and stage/worker exit codes were zero; no dependency or business-stage failure remained.

## 2026-09-11 — post-run reproducibility hardening

- Added `data.dataloader_num_workers=0` to shared PPO overrides to avoid image-dependent Ray DataLoader worker termination during tiny single-GPU runs; the completed job already passed before this hardening, so this change is queued for the next rerun if needed.
- Restored final persistent-venv `protobuf==3.20.3` after misc tooling installation, matching `training/install_requirements.sh`; import probe still passes for torch/vLLM/Transformers/Ray.
- Shell syntax and `git diff --check` were rerun. Existing historical trailing whitespace in an earlier progress entry is unchanged.

## 2026-09-11 — skill rule extraction

- Added the `Persistent environment rule` to `/data/ycfeng/codex-home/skills/stepmind-gpu-worker/SKILL.md`. It turns worker dependency failures into a CPU-host repair loop: capture the first traceback, build/pin a complete `/data` environment (including a worker-visible interpreter and offline assets), run CPU import/unit/dry-run checks, then reuse it through local NFS and verify worker versions/artifacts/status. Normal worker-side pip installation is excluded from accepted runs and retained only as bounded diagnosis.

## 2026-09-11 — reproducibility documentation hardening

- Updated `docs/03_reproduce_e2e.md` with the authoritative repeat-run recipe: prepare the persistent venv on the CPU host, run import/version and NLTK preflight, then submit exactly one H800 RJob through StepMind Python and inspect the retained work directory.
- Updated `task_memory/env_handbook.md`, `harness.md`, and `notes.md` with the worker-portable interpreter rule, pinned runtime contract, and prohibition on broad worker-side dependency installation during accepted runs.
- The documentation now points to the successful persistent-venv run (`exp-0911-152629-174180`) while retaining older CLI attempts as historical evidence. `git diff --check` passes for the documentation changes.
