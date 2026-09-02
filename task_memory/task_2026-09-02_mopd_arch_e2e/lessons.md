# Lessons

## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-03 | Recorded reusable architecture, configuration, and validation lessons from the verified run. |

1. An undeclared Hydra key must use the `+key=value` form. This applies to runtime
   extension fields such as `actor_rollout_ref.rollout.reward_mode` and
   `log_prob_top_k`.
2. The dataclass default for `RolloutConfig.log_prob_top_k` does not populate the
   dictionary read by `ray_trainer.py`; compose-time config and worker config are
   separate boundaries.
3. SFT's `sft_trainer.py` consumes the `messages` schema through
   `MultiTurnSFTDataset`; passing `prompt_key`/`response_key` from the other trainer is
   rejected and does not change the reader.
4. FSDP checkpoints need an explicit merge before vLLM evaluation. The `huggingface/`
   subdirectory in a checkpoint contains metadata and tokenizer files, not model weights.
5. MT-OPD routing is observable only when every teacher runs on every student token.
   The final run showed equal prompt shares but unequal token shares and corresponding
   M1 loss weights, closing the evidence chain for the central claim.
6. A direct runtime probe catches configuration errors earlier and more cheaply than a
   GPU run. The 15-config probe reduced all 11 cases to explicit COMPOSE/RAISE verdicts.
7. A wrapper that records every stage must preserve each exit code. Continuing for
   evidence collection is useful only when the summary still exposes failures.
