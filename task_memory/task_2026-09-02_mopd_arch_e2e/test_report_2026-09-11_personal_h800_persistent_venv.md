## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-11 | Record successful personal H800 E2E using a persistent, worker-portable Python runtime. |

# Test Report: Personal H800 Open-MOPD E2E

## Execution

Submission was performed locally on `kun-workspace-vgen2` with personal StepMind Python `RJobBackend` credentials (`STEPMIND_BACKEND=rjob`, creator `i-fengyicheng`). The launcher remained alive until terminal completion.

```bash
cd /data/ycfeng/steptron
export STEPMIND_BACKEND=rjob STEPMIND_PYTHON=/data/ycfeng/tmp/stepmind-env/bin/python
bash /data/ycfeng/tmp/mopd_personal_python.sh \
  /data/ycfeng/Open-MOPD/tests/e2e/submit_h800_image.py
```

RJob: `exp-0911-152629-174180`
Resources: `codesign`, `positive_tags=[H800]`, 1 GPU, 16 CPU, 132 GB
Image: `hub.i.basemind.com/stepmind/megatron-step:test-optimus-3.24.0-stepccl-0.0.5.post5-vllm-0.11.0.post37-20260817-2733176`
Mount: `100.96.128.191:/data/ycfeng:/data/ycfeng`
Worker Python: `/data/ycfeng/envs/openmopd-py312/bin/python` (CPython 3.12.13 runtime copied into `/data`)

## Criteria and evidence

| Criterion | Evidence | Result |
| --- | --- | --- |
| Worker allocation and GPU visibility | H800 worker `nvidia-smi`; CUDA build `12.8`; `torch.cuda.is_available()` true on worker | PASS |
| Runtime dependency contract | `torch 2.8.0+cu128`, `vllm 0.11.0`, `transformers 4.57.6`, `ray 2.55.1`, `tensordict 0.10.0`, `hydra 1.3.6`, `verl 0.7.0.dev`, `verifiable_instructions 0.1.0`, `pylatexenc`, `accelerate` imports | PASS |
| Unit and repository tests | `20_unit_tests.sh`: 133 tests; repository/eval suite: 120 passed | PASS |
| Config and launcher checks | `25_dry_runs.sh`, `15_config_probe.py` | PASS |
| SFT and merge | `30_sft.sh` exit 0; `35_merge.sh` exit 0; merged checkpoint `runs/merged/global_step_6` exists | PASS |
| RL optimizer path | `40_rl.sh` exit 0; 2 steps emitted actor loss, grad norm, reward, throughput metrics; checkpoint `runs/rl/checkpoints/global_step_2` exists | PASS |
| OPD | `50_opd.sh` exit 0; corrected invocation included explicit `ppo_micro_batch_size_per_gpu=2` | PASS |
| MT-OPD | shipped and corrected M1 runs both exit 0; per-domain shares, loss weights, conflict and actor gradient metrics emitted | PASS |
| Rollout | vLLM processed 12 local prompts and wrote `runs/eval/dummy_math_rollouts_base0_rank0.parquet` with 12 rows | PASS (technical path) |
| Scoring/verifier | scorer and verifier commands exit 0 and write `verifier_score.json` | PASS (execution) |
| Whole job terminal status | Personal API: `phase=Succeeded`, replica `SUCCEED`, worker exit marker `0`, launcher `FINAL_STATUS succeeded` | PASS |

Selected RL/MT-OPD metrics include `actor/grad_norm=0.3362`, `perf/throughput=508.17` tokens/s, `mt_opd/conflict/spread_mean=0.3826`, and distinct per-domain loss weights (`code=1.3698`, `if=0.5696`, `math=1.1210`).

## Limits and interpretation

The run uses four locally generated dummy models and synthetic rows with tiny smoke settings (1 GPU, TP/DP 1, short sequences, two optimizer steps). It exercises the repository's loss, gradient, teacher-routing, checkpoint, and vLLM control paths; it does not establish convergence, throughput, or production quality. The verifier reports `dummy_math` as `unknown_dataset` with `scored_rows=0`, so no benchmark reward or correctness claim is made. No HF weights or external datasets were downloaded. A transient Ray multiprocess cleanup traceback was present, but corrected training completed and every stage and the worker exited with code 0.

Artifacts:

- Worker log: `/data/ycfeng/tmp/openmopd-image-20260911-152629/logs/worker.log`
- Persistent run outputs: `/data/ycfeng/tmp/openmopd-image-20260911-152629/runs`
- API snapshot: `/data/ycfeng/tmp/exp-0911-152629-174180.log`
