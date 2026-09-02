# Future Work

## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-03 | Listed follow-up work that is explicitly outside this task's scope. |

* Patch `scripts/local/opd.sh` and `scripts/local/mt_opd.sh` to append
  `reward_mode` and set `log_prob_top_k` in the released launchers.
* Align teacher-0 configuration with additional teachers (`input_tokenizer=null`,
  remove-padding, and FSDP offload settings) and add positional domain-binding checks.
* Decide and document a production default for M2 reward scaling and connect
  `conflict_nats` to the conflict metric threshold.
* Fix or isolate the fake HTTP session and endpoint fixtures in `test_if_rl_grm.py`,
  then reassess the capability-subspace tolerance under the supported Torch version.
* Add a scorer fixture for `dummy_math` if a numeric verifier score is needed in future
  offline evals.
