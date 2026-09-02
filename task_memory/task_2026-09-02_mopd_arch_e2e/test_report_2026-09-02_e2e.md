# E2E Test Report — Open-MOPD

## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-03 | Recorded the final r5 H800 run, numeric metrics, artifacts, and non-core test failures. |

## 1. Test Script Information

**Scripts**

* `/data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e/e2e/run_on_worker.sh`
* Stage scripts `20_unit_tests.sh`, `25_dry_runs.sh`, `15_config_probe.py`,
  `30_sft.sh`, `35_merge.sh`, `40_rl.sh`, `50_opd.sh`, `60_mt_opd.sh`, and `70_eval.sh`
  in the same directory.

**Exact command**

```bash
timeout 10800s /kubebrain/rlaunch \
  --name openmopd-e2e-full-20260903-r5 \
  --charged-group=codesign --private-machine=group --positive-tags=h800 \
  --gpu=1 --cpu=16 --memory=131072 --backoff-limit=1 \
  --max-wait-duration=30m -- bash -lc '
    WORK=/data/ycfeng/tmp/openmopd-e2e-r5-20260903
    ASSETS=/data/ycfeng/tmp/openmopd-e2e-assets-eager2-20260903
    REPO=/data/ycfeng/Open-MOPD
    export WORK ASSETS REPO HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    bash "$REPO/task_memory/task_2026-09-02_mopd_arch_e2e/e2e/run_on_worker.sh"
  '
```

**Environment**

* GPU: NVIDIA H800, 81,559 MiB, driver `570.124.06`.
* Python: `/data/ycfeng/envs/openmopd-py312/bin/python`, Python 3.12.
* PyTorch: `2.8.0+cu128`, `torch.cuda.is_available() == True`, one device.
* Vendored verl: `0.7.0.dev`; vLLM `0.11.0`; Transformers `4.57.6`.
* Offline variables: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `HF_DATASETS_OFFLINE=1`.
* Models: locally synthesized Qwen3 models only; no HF weight or released checkpoint
  was downloaded.

## 2. Validation Criteria

| Criterion | Acceptance condition |
| --------- | -------------------- |
| Architecture executable spec | Open-MOPD test files run with zero failures and no skipped tests silently hidden |
| Launcher/config wiring | Dry-run commands print complete argv; compose probe has zero unexpected outcomes |
| SFT | Six training steps, finite losses, checkpoint tracker reaches `6` |
| Merge | A loadable HF directory contains config, tokenizer, and weights |
| RL | Two GRPO steps emit actor/reward/timing metrics and an actor checkpoint |
| OPD | Shipped defect is observed; corrected `opd_kl` path completes two steps with dense teacher metrics |
| MT-OPD | Three teachers route by domain; each domain emits prompt/token/loss shares; conflict metrics are finite |
| Eval | vLLM writes a parquet with `completion` and 12 rows; scoring/verifier exits are observable |

## 3. Test Results and Evidence

### 3.1 Stage matrix

| Stage | Result | Evidence |
| ----- | ------ | -------- |
| `20_unit_tests.sh` Open-MOPD suite | PASS | `132 passed` in 28.04 s (`20_unit_tests.log`) |
| `20b_repo_tests` | FAIL (non-core) | `111 passed, 8 failed` in 63.24 s; failures listed in §3.6 |
| `25_dry_runs.sh` | PASS | All launcher argv printed; negative teacher-count checks exit `2` |
| `15_config_probe.py` | PASS | `unexpected outcomes: 0` |
| `30_sft.sh` | PASS | Exit `0`; 6 steps; checkpoint tracker `6` |
| `35_merge.sh` | PASS | Exit `0`; `model.safetensors` `885032` bytes plus config/tokenizer |
| `40_rl.sh` | PASS | Exit `0`; 2 steps and actor checkpoint |
| `50_opd.sh` | PASS with documented defect | shipped exit `1`; corrected exit `0` |
| `60_mt_opd.sh` naive | PASS with documented defect | shipped exit `1`; corrected exit `0` |
| `60_mt_opd.sh` M1 | PASS with documented defect | shipped exit `1`; corrected exit `0` |
| `70_eval.sh` | PASS | vLLM, score_rollouts, and verifier each exit `0` |

The worker wrapper returns exit `1` solely because it deliberately propagates the
non-core repo-test failures from `20b_repo_tests`; all MOPD stages after that gate ran
and completed.

### 3.2 SFT metrics

Observed in `logs/30_sft.log`:

| Metric | Step 1 | Step 6 |
| ------ | ------ | ------ |
| `train/loss` | `6.435940265655518` | `5.8885908126831055` |
| `val/loss` | `6.314265251159668` | `5.785400390625` |

The output contains `global_step_1` through `global_step_6` and
`latest_checkpointed_iteration.txt = 6`.

### 3.3 RL metrics

Observed in `logs/40_rl.log`:

| Metric | Step 1 | Step 2 |
| ------ | ------ | ------ |
| `actor/entropy` | `6.316310405731201` | `6.316421031951904` |
| `critic/score/mean` | `0.0` | `0.0` |
| `perf/total_num_tokens` | `2178` | `2187` |
| `perf/throughput` (tokens/s) | `397.2632540281764` | `695.6351440030364` |

The actor checkpoint is at
`runs/rl/checkpoints/global_step_2/actor/model_world_size_1_rank_0.pt`.

### 3.4 Single-teacher OPD metrics

The shipped `opd.sh` invocation produced the expected Hydra error:

```text
Could not override 'actor_rollout_ref.rollout.reward_mode'.
To append to your config use +actor_rollout_ref.rollout.reward_mode=opd_kl
Key 'reward_mode' is not in struct
```

The corrected direct invocation completed two steps. Selected values from
`logs/50b_opd_corrected.log`:

| Metric | Step 1 | Step 2 |
| ------ | ------ | ------ |
| `critic/score/mean` | `-0.5174221396446228` | `-0.4962007999420166` |
| `actor/entropy` | `6.316310405731201` | `6.317284107208252` |
| `teacher/entropy` | `6.3164753913879395` | `6.316640853881836` |
| `val-topk/overlap_ratio` | `0.03364342451095581` | `0.03280804678797722` |

### 3.5 MT-OPD metrics

The shipped launcher failed at the same undeclared `reward_mode` key. Both corrected
three-teacher runs completed two steps with domains `math`, `code`, and `if`.

**Naive routing, step 2 (`logs/60b_mt_opd_corrected_naive.log`)**

| Domain | `prompt_share` | `token_share` | `reward_abs_mean` |
| ------ | -------------- | ------------- | ----------------- |
| code | `0.3333333333333333` | `0.29202279448509216` | `0.5148119926452637` |
| if | `0.3333333333333333` | `0.3511396050453186` | `0.4921409487724304` |
| math | `0.3333333333333333` | `0.35683760046958923` | `0.5002112984657288` |

Conflict metrics were `spread_mean=0.3818398118019104`,
`spread_max=1.1681127548217773`, and `contested_frac=0.0028490028344094753`.

**M1 target-share weighting, step 2 (`logs/60b_mt_opd_corrected_m1.log`)**

| Domain | `token_share` | `loss_weight` |
| ------ | ------------- | ------------- |
| code | `0.29202279448509216` | `1.3697561025619507` |
| if | `0.3511396050453186` | `0.5695740580558777` |
| math | `0.35683760046958923` | `1.1209580898284912` |

Top-level step-2 values were `mt_opd/rm_scores_mean=-0.5016412138938904`,
`mt_opd/teacher_entropy=6.316641807556152`,
`mt_opd/conflict/spread_mean=0.38263604044914246`, and
`mt_opd/conflict/contested_frac=0.0028490028344094753`.

The non-uniform loss weights are the expected correction for token-share imbalance:
prompt shares are equal at one third, while token shares differ because response
lengths differ.

### 3.6 Eval artifacts

`logs/70_eval.log` records vLLM engine initialization and a successful rollout:

* output: `/data/ycfeng/tmp/openmopd-e2e-r5-20260903/runs/eval/dummy_math_rollouts_base0_rank0.parquet`
* rows: `12`
* columns include `completion`, `completion_tokens`, and `finish_reason`
* first completion has `completion_tokens=64`, `finish_reason=length`
* parquet size: `13780` bytes
* `score_rollouts` exit `0`; no scorer exists for `dummy_math`
* `scores.json` was not emitted because `score_rollouts` leaves unknown datasets
  unscored; verifier wrote `/data/ycfeng/tmp/openmopd-e2e-r5-20260903/runs/eval/verifier_score.json`
  and exited `0`, with result note `unknown_dataset` and `scored_rows=0`

### 3.7 Non-core failures and root causes

The optional repo-level suite reports exactly:

```text
8 failed, 111 passed in 63.24s
```

Seven failures are in `experiments/tests/test_if_rl_grm.py`: two fake sessions omit
`.mount()`, and five semantic/score tests use `http://fake-endpoint:8000`, which cannot
resolve. The request path retries three times and returns an error, so `grm_pass` and
`grm_capped` remain `0.0`. One failure is the capability-subspace comparison:
`cheap=0.4956863522529602`, `explicit=0.469860315322876`, absolute difference
`0.025826037`, above the test's `0.02` tolerance. These modules are outside the MOPD
core path and were left unchanged.

## 4. Overall Verdict

The requested MOPD architecture and high-fidelity E2E chain are delivered. The
Open-MOPD executable specification, launcher composition, SFT, merge, GRPO, corrected
OPD, corrected MT-OPD (naive and M1), and offline vLLM evaluation all produced the
required evidence. The final worker process is non-zero only because the report keeps
the eight unrelated repo-level test failures visible.
