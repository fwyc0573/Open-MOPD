# Future Work

## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-03 | Listed follow-up work that is explicitly outside this task's scope. |
| 2026-09-04 | Double-checked all five items; landed scoped fixes and recorded deferred design choices. |

* **Closed:** `scripts/local/opd.sh` and `scripts/local/mt_opd.sh` append
  `+reward_mode` and set `+log_prob_top_k=256`; Hydra probe and launcher dry-run pass.
* **Closed for the supported launcher path:** teacher-0 now matches additional teachers
  (`input_tokenizer=null`, remove-padding, FSDP parameter offload). Trainer setup validates
  positional count, unique/non-empty labels, and rejects unknown batch domains before routing.
* **Partially closed:** `conflict_nats` now reaches the conflict metric as well as M3 policy.
  M2's `divide` -> `multiply` default switch remains deferred because it changes production
  optimization semantics; source comments and tests support `multiply` as the recommended
  explicit setting, but do not establish a backward-compatible default migration.
* **Closed:** GRM tests now stub `_request_completion` (the seam used by concurrent fan-out)
  and FakeSession implements `.mount()`. Capability explicit directions use deterministic
  exact SVD; GRM has `17 passed` and capability tests have `69 passed` under the supported
  Torch environment.
* **Deferred:** `dummy_math` has no production scorer registry entry. The current offline
  eval intentionally records `unknown_dataset`/`scored_rows=0`; add a task-local fixture only
  when a numeric dummy verifier score is required by a future eval experiment.
