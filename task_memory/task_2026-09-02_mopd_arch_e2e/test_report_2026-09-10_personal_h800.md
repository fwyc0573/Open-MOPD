## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-10 | Record current personal H800 platform evidence and pending full E2E. |

# Personal H800 reproduction — in progress

## Execution and environment

The local submission interpreter is `/data/ycfeng/tmp/stepmind-env/bin/python` (Python 3.10.6, brainpp 2.8.6, no conda activation). Personal credentials are loaded before creating any client, following `/data/ycfeng/stepfun-env-handbook/stepmind-python-rjob.md` section 2. Values are never archived.

After that initialization, reproduce submission with:

```bash
"$STEPMIND_PYTHON" -u /data/ycfeng/Open-MOPD/tests/e2e/submit_h800_image.py
```

Worker command:

```bash
bash /data/ycfeng/Open-MOPD/tests/e2e/run_h800_image.sh
```

The Python launcher generates a fresh WORK, requests H800/codesign, and calls spawn_tasks with cwd and code_mount_point both `/data/ycfeng`. The new job is `exp-0910-224825-381481`; output directory is `/data/ycfeng/tmp/openmopd-image-20260910-224825`. The API confirms creator `i-fengyicheng`, NFS `100.96.128.189:/data/ycfeng:/data/ycfeng`, 1 GPU, 16 CPU, and 132000Mi memory. Company image:

```text
hub.i.basemind.com/stepmind/megatron-step:test-optimus-3.24.0-stepccl-0.0.5.post5-vllm-0.11.0.post37-20260817-2733176
```

## Criteria and observations

| Criterion | Observed result | Verdict |
| --- | --- | --- |
| Personal auth, local NFS, H800 allocation | Runtime probe exp-0910-222602-152735: creator i-fengyicheng, H800 node gpu-h800-0116, correct NFS annotation, Succeeded | PASS for platform/runtime probe only |
| CUDA runtime import | /usr/bin/python, torch 2.8.0+cu128, CUDA True, vllm 0.11.0.post37 | PASS for inspected imports only |
| Old setup command | uv absent; 00_setup_env.sh line 28 exits 127 | FAIL; image runner uses image Python and separately resolves torchrun |
| Full E2E job | 15:27:51 UTC: Pending, reason RJob is queuing; codesign-default ranking 3, H800 remaining 0 | Pending |
| Dummy assets, tests, SFT, merge, RL, OPD, MT-OPD naive/M1, eval | No fresh business evidence from current job yet | Pending |

The runtime probe used `|| true` to collect multiple observations, so Succeeded alone cannot establish import success. The explicit torch/CUDA/vLLM output above establishes only those imports. It does not prove the full verl dependency set or training behavior.

The image vLLM post release differs from historical vLLM 0.11.0. This is an execution compatibility question to settle with the actual full chain; no numeric equivalence to historical r5 is claimed.

## Known diagnostic corrections

Zero log rows from the local API did not mean that no worker output existed. A bounded read-only SSH log query exposed `uv: command not found` for exp-0910-221824-700603. Earlier inferred mount explanations for already-correct annotations were unsupported. Direct rjob_run also skipped spawn_tasks mount setup in exp-0910-220533-432310; the new launcher uses spawn_tasks as required.

The asset generator accepts `--out`, not `--output`. The image launcher invokes the actual supported flag. NFS annotation source follows submission cwd, while code_mount_point is the worker target.

## Acceptance still outstanding

Verify every stage's logs and artifacts, including training losses/checkpoints, OPD reward statistics, three MT-OPD domain shares/weights and finite conflict metrics, evaluation parquet/completions and scores. Preserve any failure and its traceback. A job name or terminal process status cannot replace this audit.

Launcher verification: bash -n passed for tests/e2e/run_h800_image.sh and the existing e2e/_common.sh; Python AST parsing passed for tests/e2e/submit_h800_image.py; git diff --check passed. The exact-job API verifies the actual submitted creator/mount/resource payload. These checks validate submission configuration; the worker business run is still pending.
