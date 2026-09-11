## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-11 | Audited E2E stage chaining, reward dispatch, rollout/eval selection, and smoke reductions. |

# E2E standardness audit

The current worker scripts exercise the repository entry points, but several choices mean
the run is a path/smoke check rather than a standard training progression.

1. **Stage model chaining is missing.** `e2e/_common.sh:67` fixes
   `actor_rollout_ref.model.path` to the raw dummy student for every PPO stage. Thus
   `40_rl.sh`, `50_opd.sh`, and `60_mt_opd.sh` do not consume the SFT merged model or
   preceding RL/OPD checkpoints. `docs/01_architecture.md:363-365` documents the required
   FSDP-to-HF conversion at each boundary. Loss and gradient observations therefore use an
   untrained initialization rather than the intended SFT -> RL -> OPD progression.

2. **Evaluation defaults to the raw student.** `e2e/70_eval.sh:15-17` says a merged
   checkpoint should be selected when available, but the implementation always assigns
   `EVAL_MODEL=$MODEL_STUDENT` unless the caller exports `EVAL_MODEL`. The vLLM rollout
   consequently does not evaluate the trained checkpoint by default.

3. **Synthetic code/IF rows do not reach their verifiers.**
   `e2e/10_make_dummy_assets.py:181-184` emits `data_source=dummy_code` and
   `dummy_if`. `training/verl/verl/utils/reward_score/opd_val_dispatch.py` recognizes
   production names such as `livecodebench*`, `ifeval`, `ifbench`, and `nemotron_if`; the
   dummy names fall through to the math scorer. Their validation/custom rewards therefore
   do not represent code or instruction-following scoring. The rows also lack official LCB
   testcase / IF constraint metadata. Teacher domain routing is exercised, but mixed-domain
   reward semantics are not.

4. **Verifier status does not gate the E2E exit code.** `e2e/70_eval.sh:73-86` captures
   `rc3` from `evals.verifier.score` but exits using only `rc1 || rc2`. A verifier failure,
   unknown dataset, or missing verifier output can therefore coexist with a reported stage
   success.

5. **Intentional smoke reductions limit numerical claims.** `_common.sh:37-48` uses
   prompt/response limits 128/64, batch 12, micro-batch 2, two responses, and two steps;
   SFT further uses batch 4 and max length 256 (`30_sft.sh`). `use_remove_padding=False`
   and eager attention (`_common.sh:54-69`) avoid flash-attn and exercise a different kernel
   path. These settings are suitable for reachability and finite loss/gradient checks, but
   not convergence, production loss, throughput, or kernel-equivalence claims.

6. **Failure continuation can hide invalid downstream dependencies.**
   `run_on_worker.sh:19-31` intentionally continues after each stage failure. This is useful
   for collecting independent diagnostics, but a later stage passing cannot establish a valid
   predecessor checkpoint. The final summary must be interpreted per-stage and the wrapper
   failure retained.


7. **Offline dataset guard differs by entry point.** The GPU wrapper `tests/e2e/run_h800_image.sh:8` exports `HF_DATASETS_OFFLINE=1`, but `_common.sh` itself only exports Hub and Transformers offline flags (`:17-18`). Running `e2e/run_all.sh` directly therefore does not enforce the harness's no-network dataset requirement (`harness.md:12`).

## Implication for a standard E2E claim

To claim standard training, the driver would need to (a) merge SFT and point RL at that merged
student, (b) run and merge each domain RL teacher before OPD/MT-OPD, and (c) evaluate the
chosen final merged checkpoint. This is a broader workflow change than dependency repair.
A safe report for the current run is therefore “repository path and interface smoke test on
local dummy assets”; do not present its losses, gradients, rewards, or eval score as a
standard training result.
