# Requirements — Open-MOPD Architecture Deconstruction + E2E Test

## Modification History

| Date       | Summary of Changes                                     |
| ---------- | ------------------------------------------------------ |
| 2026-09-10 | Record personal Python H800 reproduction and verified runtime observations. |
| 2026-09-02 | Initial capture of the original request (two deliverables) |

## [Original Request]

> 这是一个new task，
>
> 1. 对该repo的核心mopd的codebase的架构结构进行剖析，帮助我充分理解和解构它们，你需要向我清晰地描述 项目核心模块的架构组织，调用逻辑，输入和输出。将这些内容落地为docs指导我我阅读和理解。
> 2. 进行repo的测试试验，完成完整的e2e流程测试。如果需要使用model，请暂时使用dummy format来避免权重下载和使用。清晰地记录整个流程，执行shell和cmds，我需要根据你的guidance doc进行人工复现。

## Derived Requirements

### R1 — Architecture deconstruction docs

- R1.1 Describe the **architecture organization** of the core modules (what module owns what).
- R1.2 Describe the **call logic** (control flow / invocation chain across stages and inside the MOPD step).
- R1.3 Describe the **inputs and outputs** of each core module (data schema, tensor keys, config keys, artifacts).
- R1.4 Land all of the above as **docs** that guide reading and understanding of the codebase.

### R2 — E2E pipeline test

- R2.1 Run a **complete end-to-end flow** test of the repo.
- R2.2 Where a model is needed, use a **dummy model format** — no HF weight download, no released checkpoint use.
- R2.3 Record the **whole flow, the shell and the commands** executed.
- R2.4 The record must be a **guidance doc that the user can manually reproduce** step by step.

## Open Questions (resolved in notes.md / issues.md as they close)

- Q1 GPU scope for R2: this machine (`nvidia-smi` absent, `torch.cuda.is_available()==False`)
  cannot run vLLM rollout or FSDP training. Decide between CPU-only logic-level E2E and
  an `rlaunch` GPU-worker E2E.

## [Original Request] — 2026-09-10 continuation

> 复现任务/data/ycfeng/Open-MOPD/task_memory/task_2026-09-02_mopd_arch_e2e，使用gpu worker h800集群（注意与h200的集群的参数使用区别）

> 已经为当前machine配置了操作gpu worker的新的skill，更新了相关docs。请你参考这些补充的docs继续上述复现任务

> 请你再次提交测试，并返回给我需要我观测的 rjob的id

Decision: submit locally as i-fengyicheng through StepMind Python spawn_tasks/RJobBackend, mount local source only, use H800/codesign, and retain the complete offline dummy E2E scope. Report each actual job ID and observed status.
