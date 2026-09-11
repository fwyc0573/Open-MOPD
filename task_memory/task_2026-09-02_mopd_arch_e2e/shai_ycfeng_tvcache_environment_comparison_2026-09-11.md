## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-11 | Compared the successful TVCache H800 reproduction environment with Open-MOPD worker runs. |

# Environment comparison

The successful TVCache run on `shai-ycfeng` is not using the same image or runtime model as the Open-MOPD run.

## TVCache successful H800 path

Read-only inspection of:

```text
/data/ycfeng/stepfun-performance-optimization/new-topic-research/related-work/TVCache/.worktrees/tvcache-rl-reproduction
```

The accepted H800 rollout uses:

```text
image: hub.stepfun-inc.com/stepcast/stepcast:vllm-openai-v0.19.0
GPU: 2 H800
charged_group: codesign
Python environments:
  /data/ycfeng/tmp/tvcache-train-py312
  /data/ycfeng/tmp/tvcache-videoagent-20260909
  /data/ycfeng/tmp/tvcache-videollava-20260909
```

The worker does not install a broad Python requirements set at startup. It reuses prebuilt, persistent environments and pinned offline HF assets. The script explicitly sets `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `HF_HUB_OFFLINE=1`, and process-local CUDA library paths. It launches VideoAgent, Video-LLaVA, and TVCache server with their dedicated interpreters.

The successful acceptance scope is real inference-driven rollout for one fixed EgoSchema video, four provider rollouts, tool execution, cache hit/miss/fork metrics, and teardown. It explicitly does not execute an RL optimizer update.

## Open-MOPD current path

Open-MOPD uses:

```text
image: hub.i.basemind.com/stepmind/megatron-step:test-optimus-3.24.0-stepccl-0.0.5.post5-vllm-0.11.0.post37-20260817-2733176
GPU: 1 H800 or H200
Python: /usr/bin/python (system image runtime)
Torch: 2.8.0+cu128
vLLM: 0.11.0.post37
```

The image contains Torch/vLLM/Transformers but does not contain the full `verl` development/runtime dependency set. The current worker wrapper therefore installs missing libraries into a per-run `$WORK/site` directory. The first broad installation accidentally upgraded `transformers` to `5.17.0`, which conflicted with the image's older `huggingface_hub`; subsequent iterations switched to `--no-deps` and preserved the image versions.

## Root cause of the apparent discrepancy

The TVCache task is successful because its durable task recipe assembled and validated dedicated environments before the GPU run. The worker only executes the already prepared interpreters and pinned model/cache assets. Open-MOPD is currently bootstrapping `verl` dependencies inside a minimal serving/training image at runtime. These are materially different environment contracts, so the missing-library sequence is expected and does not imply that the H800 platform itself is inconsistent.

The two tasks also have different acceptance scopes. TVCache's accepted result is a real provider rollout without optimizer updates. Open-MOPD requires SFT, merge, RL, OPD, MT-OPD, and eval, which exercises a much larger dependency graph (`hydra`, `tensordict`, `torchdata`, `peft`, `accelerate`, `ray`, `codetiming`, `pylatexenc`, IF-RL packages, and related libraries).

This scope distinction explains why TVCache did not encounter the Open-MOPD `verl` training import chain: its accepted H800 run starts three already-built service processes and calls the provider rollout driver. It does not import `verl.trainer.main_ppo`, instantiate FSDP actor/reward configs, or execute SFT/RL/OPD/MT-OPD trainers. Comparing its short dependency list with Open-MOPD's full training dependency graph would therefore be misleading.

## Operational lessons adopted

1. Do not install unpinned latest packages into the system image. Preserve image Torch, vLLM, Transformers, and Hugging Face Hub versions.
2. Use a persistent, prebuilt Open-MOPD environment or a fully pinned per-run wheel set before allocating a long GPU job.
3. Keep offline model/data assets separate from Python dependency preparation.
4. Treat a successful serving rollout as evidence of serving/runtime compatibility only; it does not establish the full training dependency contract.

The detailed Open-MOPD semantic limitations remain in [standardness_audit_2026-09-11.md](standardness_audit_2026-09-11.md). The current H800 run remains a diagnostic E2E until its dependency and configuration failures are repaired.
