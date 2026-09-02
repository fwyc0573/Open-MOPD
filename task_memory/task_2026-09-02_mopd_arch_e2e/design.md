# Task Design

## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-03 | Recorded the architecture-document and E2E verification design used for this task. |

## Objective

Deliver two independent but connected artifacts: a code-grounded architecture reading
guide, and a reproducible high-fidelity MOPD pipeline run.

## Design Decisions

1. Keep architecture claims in `docs/01_architecture.md` and
   `docs/02_mt_opd_algorithm.md`; every load-bearing claim carries a source
   `file:line`.
2. Keep all executable reproduction assets under this task directory. The scripts call
   released entrypoints and label known launcher defects before invoking corrected
   direct commands.
3. Use locally synthesized Qwen3 models with distinct seeds. This exercises tokenizer,
   FSDP, vLLM, teacher routing, and checkpoint conversion without network access.
4. Treat `reward_mode` and `log_prob_top_k` as runtime extension keys. Hydra composition
   appends them with `+`; `ray_trainer.py` consumes and removes `reward_mode` before
   worker dataclass conversion.
5. Preserve failures from unrelated repository tests in the report. The core stage
   evidence remains independently inspectable in its own logs.

## Boundary Contracts

* SFT input: parquet `messages` rows; output: FSDP `global_step_N` shards.
* Merge input: FSDP checkpoint plus `huggingface/`; output: `model.safetensors` and
  tokenizer/config directory.
* PPO input: RL parquet with `domain` for MT-OPD; output: per-step metrics and actor
  checkpoints.
* MT-OPD teacher path: `[B,T,K]` teacher log-prob tensors are routed by domain and
  overwritten into the existing single-teacher reward key.
* Eval input/output: parquet prompts -> parquet completions -> optional scorer JSON.
