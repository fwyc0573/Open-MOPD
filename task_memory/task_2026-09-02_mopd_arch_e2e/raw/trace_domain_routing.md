# Domain label → per‑domain loss weight: full trace

## 0. Upstream vs Open‑MOPD

| Component | Origin |
|---|---|
| `verl/utils/dataset/rl_dataset.py` `RLHFDataset`/`collate_fn` | **upstream verl** — contains *zero* `domain` logic (`grep -n domain` on the file returns nothing). It transports `domain` only because it is column‑agnostic. |
| `verl/workers/actor/mt_opd.py` (whole file, 562 lines) | **Open‑MOPD** |
| `ray_trainer.py:76-83, 86-92, 659-777, 1145-1165, 1427-1440, 1484-1497, 1787-1868, 2100-2230` | **Open‑MOPD** patches |
| `dp_actor.py:601-774` (`compute_distillation_reward`), `972-976`, `1095-1132` | **Open‑MOPD** (OPD reward + M1/M4 hooks) |
| `experiments/data/build_mix_rl_raw_union.py`, `build_mix_rl_union.py` | **Open‑MOPD** |
| `core_algos.py:876-902` `token_reward_direct` | **Open‑MOPD** (registered estimator, Chinese docstring, dense 3‑D path) |

---

## 1. The parquet column

Top‑level **`domain`** (Arrow `string`), plus a redundant copy inside `extra_info.domain`:

- `/data/ycfeng/Open-MOPD/experiments/data/build_mix_rl_raw_union.py:64` — `pa.field("domain", pa.string())` in `OUTPUT_SCHEMA`
- `:52` — `pa.field("domain", pa.string())` inside `EXTRA_INFO_TYPE`
- `:95` and `:108` — both written per row (`normalized_extra["domain"]`, `record["domain"]`)
- `:33` — `DOMAINS = ("math", "code", "if")`
- Same pair in `experiments/data/build_mix_rl_union.py:104` (`extra["domain"]`) and `:108` (`record["domain"]`)

On‑disk row schema: `data_source: string, prompt: list<struct<role,content>>, ability: string, reward_model: struct<style,ground_truth>, extra_info: struct<... domain ...>, domain: string` (`build_mix_rl_raw_union.py:57-66`).

**Only the top‑level `domain` is ever read by the trainer.** `extra_info.domain` is dead weight for routing purposes (nothing in `verl/` reads it — the trainer's only reads are `non_tensor_batch["domain"]` at `ray_trainer.py:1818, 1859, 2109`).

---

## 2. RLHFDataset → `non_tensor_batch["domain"]`

There is **no explicit handling** — it is pure pass‑through:

1. `rl_dataset.py:200` `_load_parquet_dataset(parquet_file)` → `:202` `datasets.concatenate_datasets(...)`, so `domain` becomes a HF dataset column.
2. `rl_dataset.py:338` `row_dict: dict = self.dataframe[item]` — the **whole** row, `domain` included; `:497` `return row_dict`. Nothing pops or renames it.
3. `rl_dataset.py:83-111` `collate_fn`: `domain` is not a `torch.Tensor`, so it lands in `non_tensors` (`:102-103`) and at `:109` becomes `np.fromiter(val, dtype=object, count=len(val))` → `np.ndarray(dtype=object)` of shape `(B,)`.
4. Wired at `ray_trainer.py:828-831` (`collate_fn = default_collate_fn`) → `:836-842` `StatefulDataLoader(..., collate_fn=collate_fn, sampler=train_sampler)`.
5. `ray_trainer.py:1881` `batch = DataProto.from_single_dict(batch_dict)`; `protocol.py:492-498` routes every `np.ndarray` into `non_tensors`. Result: **`batch.non_tensor_batch["domain"]`, object array, len B.**

Optional consumer of the *same* column at dataset level: `verl/utils/dataset/domain_weighted_sampler.py:58-66` requires `dataframe.column_names` to contain `"domain"` and builds per‑domain index pools. It is reachable only via `data.sampler.class_path` (`main_ppo.py:440-449`) and is **not wired by any released script** (only `task_memory/.../20_unit_tests.sh:23` mentions it).

---

## 3. Surviving rollout

`_get_gen_batch` (`ray_trainer.py:1145-1165`) is the Open‑MOPD patch that protects it:

```python
1149  reward_model_keys = (
1150      {"data_source", "reward_model", "extra_info", "uid", "domain"} & batch.non_tensor_batch.keys()
1151  )
1155  non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
1162  if self.async_rollout_mode:
1163      gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
```

So `domain` is **never popped from `batch`**; in async mode a copy also rides on `gen_batch`. Then:

- `ray_trainer.py:1892-1894` `gen_batch_output = gen_batch.repeat(rollout.n, interleave=True)`
- `ray_trainer.py:1940-1941` `batch = batch.repeat(rollout.n, interleave=True)` then `batch = batch.union(gen_batch_output)`. Duplicate non‑tensor keys are *asserted equal*, not overwritten (`protocol.py:187-198` `union_numpy_dict`), so the async duplicate is safe.
- `ray_trainer.py:1950` `_balance_batch` reorders via `protocol.py:1035-1041 reorder`, which reorders `non_tensor_batch` with the same index array → `domain` stays row‑aligned with the tensors.

Note on async: `agent_loop.py:778-802` builds the returned `non_tensor_batch` only from `__num_turns__`, `reward_extra_info`, `multi_modal_inputs` and `extra_fields`. `domain` reaches the loop as a per‑row kwarg (`agent_loop.py:484`) but is **not** copied into `extra_fields` by any loop I found, so in practice the surviving copy is the one on `batch`, not on `gen_batch_output`. The test at `test_mt_opd_batch_routing.py:210-223` synthesizes the *other* case (rollout carries `domain`) and verifies the union does not corrupt it.

---

## 4. `mt_opd.teacher_domains` config → index order

Hydra key paths (all read in `ray_trainer.py:659-777`, only when `reward_mode == "mt_opd"`):

| Key path | Line | Attribute |
|---|---|---|
| `mt_opd.teacher_domains` (list[str]) | 662 | `self.mt_teacher_domains`; **`:665-666` raises if empty** |
| `mt_opd.n_additional_teachers` (int) | 663 | `self.mt_n_additional` |
| `mt_opd.domain_weighting` (`domain_routing`\|`uniform`) | 664 | `self.mt_domain_weighting`, default `"domain_routing"` |
| `mt_opd.target_share_domains` + `mt_opd.target_share_values` (parallel lists) | 675-688 | `self.mt_target_gradient_shares` |
| `mt_opd.target_gradient_shares` (dict, fallback) | 690-695 | same |
| `mt_opd.normalize_reward_scale` (bool\|float α∈[0,2]) | 706-717 | `self.mt_normalize_reward_scale` |
| `mt_opd.reward_scale_stat` (`mean\|q75\|q90\|top10`) | 718 | |
| `mt_opd.reward_scale_direction` (`divide\|multiply`) | 719-726 | |
| `mt_opd.reward_scale_anchored` (bool) | 727-738 | |
| `mt_opd.conflict_policy` (`none\|mask\|consensus`) | 746-753 | |
| `mt_opd.conflict_nats`, `mt_opd.conflict_quantile` | 754-765 | |
| `mt_reward_model_{1..N}.*` overlay | 1429-1440 | merged onto `reward_model`, one worker each |
| `actor_rollout_ref.rollout.reward_mode=mt_opd` | 88 (`rollout.get("reward_mode", "opd_kl")`), then **deleted** from the config at 89-91 | |

Validation at `:696-702`: `target_gradient_shares` keys must be a subset of `teacher_domains`, else `ValueError`. **There is no validation that the batch's `domain` values are a subset of `teacher_domains`** — see §8.

Teacher ordering is positional: teacher 0 = `reward_model.model.path` (`self.rm_wg`), teachers 1..N = `mt_reward_model_i` (`ray_trainer.py:1427-1440` register, `1492-1497` `self.mt_rm_wgs`). `scripts/local/mt_opd.sh:57-67` builds `domain_list` from `LOCAL_DOMAINS` (default `math,code,if`, `scripts/local/common.sh:31`) and `:58-59` hard‑fails on a domain/teacher count mismatch, but the element‑for‑element correspondence between `--teacher` order and `--domains` order is a **convention only** (documented in `task_memory/task_2026-09-02_mopd_arch_e2e/e2e/60_mt_opd.sh:18-19`), unenforced.

### String → index

`ray_trainer.py:2109-2133`:

```python
2109  domains = batch.non_tensor_batch.get("domain", None)
2110  if domains is None: raise ValueError(...)          # see §8
2116  teacher_logps = [batch.batch["teacher_on_student_log_probs"]]
2118  while f"mt_teacher_{i}_on_student_log_probs" in batch.batch.keys(): teacher_logps.append(...)
2121  if len(teacher_logps) != len(self.mt_teacher_domains): raise ValueError(...)
2128  domain_weights = build_domain_weights(domains, self.mt_teacher_domains,
                                           device=teacher_logps[0].device,
                                           weighting=self.mt_domain_weighting)
```

`mt_opd.py:24-62` — signature `build_domain_weights(domains, domain_order, device, weighting="domain_routing") -> torch.Tensor [B, N] float32`:

```python
55  domain_to_idx = {d: i for i, d in enumerate(domain_order)}
57  teacher_idx = domain_to_idx.get(str(d), -1)
58  if teacher_idx >= 0: weights[sample_idx, teacher_idx] = 1.0
60  else:               weights[sample_idx] = 1.0 / n     # unknown label → uniform
```

`weighting == "uniform"` (`:51-52`) short‑circuits to `1/N` for every row.

---

## 5. Routed teacher → `rm_scores`

`mt_opd.py:65-87` `select_routed_teacher_logprobs(teacher_logprobs: list[[B,T,K]], domain_weights [B,N]) -> [B,T,K]`: stacks to `[N,B,T,K]` (`:84`), reshapes weights to `[N,B,1,1]` (`:86`), returns `(w * stacked).sum(0)`. One‑hot rows → exact selection; uniform rows → **arithmetic mean of log‑probs** (geometric mean in prob space, unnormalized). `test_mt_opd_domain_metrics.py:160-186` pins that this does **not** alias/mutate `teacher_logps[0]`.

Then:

```python
2140  routed_lp, conflict_token_weight, _cm = apply_teacher_conflict_policy(...)   # M3
2149  batch.batch["teacher_on_student_log_probs"] = routed_lp     # <-- the routing handoff
2151  distillation_output = self.actor_rollout_wg.compute_distillation_reward(batch)
2153  batch = batch.union(distillation_output)
```

`dp_actor.compute_distillation_reward` (`dp_actor.py:601`) reads exactly that key at `:663` (`T_on_S = data.batch["teacher_on_student_log_probs"]`) and for `top_k_strategy="only_stu"` computes at `:710-714`:

```python
kl_val = S_logp - T_on_S
norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
rm_scores = -kl_val * norm_weights           # [B, T, K]
```

returned as `res_tensors["rm_scores"]` (`:773-774`). Teacher log‑probs themselves come from `fsdp_workers.py:2664` `compute_rm_score` (per teacher), unioned per teacher as `mt_teacher_{i}_on_student_log_probs` at `ray_trainer.py:2102-2107`.

`rm_scores` → advantage: `ray_trainer.py:2350-2355` `should_reuse_inline_reward` returns **False** for mt_opd (`use_rm=True`, `reward.py:110-112`), so `:2378 compute_reward(batch, self.reward_fn)` runs and the naive manager short‑circuits at `naive.py:126-131` returning `data.batch["rm_scores"]` verbatim → `:2473 batch.batch["token_level_scores"]` → `:2499 token_level_rewards` → `:2620 compute_advantage(adv_estimator="token_reward_direct")` → `core_algos.py:895-902` `advantages = token_level_rewards * response_mask.unsqueeze(-1)` → `batch.batch["advantages"]` `[B,T,K]`.

---

## 6. Per‑domain metrics

`ray_trainer.py:2156-2183`:

- `metrics["mt_opd/rm_scores_mean"]`, `metrics["mt_opd/teacher_entropy"]` (2156-2162)
- `compute_domain_share_metrics(domains, response_mask, rm_scores=batch.batch["rm_scores"])` (2170-2176) → `mt_opd.py:90-156`, emits per **observed** label `dom`:
  `mt_opd/domain/{dom}/prompt_share`, `/token_share`, `/prompt_count`, `/mean_response_length`, `/reward_abs_mean` (`:148-155`). Raises if `response_mask.shape[0] != len(labels)` (`:119-123`); returns `{}` for `None`/empty (`:116-117`).
- `compute_teacher_conflict_metrics(teacher_logprobs, response_mask, domains)` (2177-2183) → `mt_opd.py:494-562`: `mt_opd/conflict/spread_mean|contested_frac|spread_max` plus `mt_opd/conflict/{dom}/spread_mean|contested_frac` (`:540-561`); `{}` when `< 2` teachers (`:525-526`).

Note the metric key uses the **raw observed label**, not a registry entry — a typo'd domain silently creates a new metric series.

---

## 7. Loss weights → actor loss

`ray_trainer.py:2189-2230`:

```python
2189  if self.mt_reward_scale_anchored and self._mt_reward_anchor is None:
2197      self._mt_reward_anchor = {str(d): metrics[f"mt_opd/domain/{d}/reward_abs_mean"] ...}
2202  dom_w = compute_domain_loss_weights(
            domains=domains, response_mask=response_mask,
            target_shares=self.mt_target_gradient_shares,
            rm_scores=batch.batch["rm_scores"],
            normalize_reward_scale=self.mt_normalize_reward_scale,
            reward_scale_stat=self.mt_reward_scale_stat,
            reward_scale_direction=self.mt_reward_scale_direction,
            reward_anchor=self._mt_reward_anchor if self.mt_reward_scale_anchored else None)
2218  if conflict_token_weight is not None:                     # M3 mask folds in
2219      per_seq = (conflict_token_weight * response_mask).sum(-1) / response_mask.sum(-1).clamp(min=1.0)
2222      dom_w = per_seq if dom_w is None else dom_w * per_seq
2224  batch.batch["domain_loss_weight"] = dom_w                 # [B] float32
2228  metrics[f"mt_opd/domain/{dom}/loss_weight"] = dom_w[sel[0]].item()
```

`mt_opd.py:159-335` `compute_domain_loss_weights(...) -> torch.Tensor [B] | None`:
- `:207-208` returns `None` if `target_shares is None` or no domains → default path untouched.
- `:221-229` targets renormalized over domains **present in this batch**; all‑zero → equal split (`:226-227`).
- `:266-288` per domain scale = `target[d] / observed_token_share[d]`.
- `:301-326` optional M2 magnitude correction (`divide` = `sc / mag**α`, `multiply` = `sc * clamp((mag/ref)**α, 0.05, 20)`).
- `:333-334` token‑weighted mean normalized to 1, so total loss scale (and effective LR) is unchanged.

Into the actor:

- `dp_actor.py:975-976` — `domain_loss_weight` added to `select_keys` (it is a *tensor*, so it survives `prepare_dynamic_batch` reordering; the comment at `:972-974` and `ray_trainer.py:2186-2188` state this is exactly why a tensor was chosen over the string labels).
- `dp_actor.py:1038` `advantages = model_inputs["advantages"]`
- `dp_actor.py:1128-1132`:
  ```python
  if "domain_loss_weight" in model_inputs:
      _dw = model_inputs["domain_loss_weight"].to(advantages.dtype)
      advantages = advantages * _dw.view(-1, *([1] * (advantages.dim() - 1)))
  ```
  Applied **after** the forward pass and after M4 `refresh_opd_advantage` (`:1095-1108`), and to `advantages` rather than `response_mask` so `ppo_kl`/`pg_clipfrac` stay unweighted (`:1116-1127`). Equivalence pinned by `tests/test_mt_opd_gradient_share.py:132-155`.
- `dp_actor.py:1179-1188` `policy_loss_fn(..., advantages=advantages, response_mask=response_mask, loss_agg_mode=...)` → `pg_loss` → `:1216 loss.backward()`.
- Driven from `ray_trainer.py:3527 self.actor_rollout_wg.update_actor(batch)`.

---

## 8. Edge cases

### Domain string not in `teacher_domains`

Two independent, **non‑failing and inconsistent** behaviors:

1. **Teacher routing** — `mt_opd.py:57-61`: `domain_to_idx.get(str(d), -1)` misses → `weights[sample_idx] = 1.0 / n`, i.e. the sample is distilled against the *log‑prob mean of all N teachers*. Silent, no warning, no metric.
2. **Loss weighting** — `mt_opd.py:225`: `raw = {d: float(target_shares.get(d, 0.0)) for d in present}`. An unknown label present in the batch gets target `0.0`, so `:288` stores scale `0.0 / observed == 0.0`, `:328-330` sets `weights[i] = 0.0`, and `:333-334` rescales the *remaining* domains up. **Those sequences contribute zero gradient.** This only bites when `mt_opd.target_gradient_shares` / `target_share_domains` is set; with it unset `dom_w is None` (`:207`) and the sample trains normally with the uniform‑teacher target from (1).

Both paths also mint fresh metric series `mt_opd/domain/<unknown>/*` (`mt_opd.py:140-155`). The guard at `ray_trainer.py:696-702` only checks config‑vs‑config; nothing checks data‑vs‑config. The e2e script's header comment (`task_memory/task_2026-09-02_mopd_arch_e2e/e2e/60_mt_opd.sh:15-16`) claims "`ray_trainer.py:2100-2124` raises … if the label set does not fit teacher_domains" — **that claim is wrong**; lines 2100-2126 only check presence of the key and the teacher *count*.

Separately, the DAPO domain‑balance filter hard‑codes `("math","code","if")` and skips unknown labels: `ray_trainer.py:1820-1831` (`if uid in seen or domain not in selected_by_domain: continue`) and `:1863-1868`. That is a second, independent classification site not driven by `teacher_domains` — active only when `reward_model.reward_manager == "mixed"` (`:1787-1789`).

### Missing `domain` key

- `reward_mode == "mt_opd"`: fails loud and early, `ray_trainer.py:2109-2114`:
  ```python
  domains = batch.non_tensor_batch.get("domain", None)
  if domains is None:
      raise ValueError("reward_mode=mt_opd requires non_tensor_batch['domain'] on the training batch")
  ```
  This fires **after** all N teacher forward passes (`:2102-2107`), so a missing column wastes one full teacher pass before erroring.
- Elsewhere it degrades silently: `_get_gen_batch`'s set intersection (`:1150`) just yields a smaller set; the DAPO filter falls back to uid‑only selection (`:1860-1861`, `:1819`); `compute_domain_share_metrics`/`compute_domain_loss_weights` return `{}` / `None` (`mt_opd.py:116-117`, `:207-208`).
- `DomainWeightedSampler` fails at construction: `domain_weighted_sampler.py:58-59` raises `"DomainWeightedSampler requires a top-level dataset 'domain' column"`.

---

## 9. Cross‑check against `tests/trainer/ppo/test_mt_opd_batch_routing.py`

| Test | Line | Verifies against source |
|---|---|---|
| `test_get_gen_batch_preserves_mt_opd_domain_for_teacher_routing` | 46-58 | `batch.non_tensor_batch["domain"]` survives `_get_gen_batch` in both modes; asserts `"domain" not in gen_batch.non_tensor_batch` for **sync** and present for **async** — exactly matches `ray_trainer.py:1150` ∩ `1155` vs `1162-1163`. |
| `test_mt_opd_domain_survives_async_rollout_union_and_repeat` | 210-237 | `repeat(1) + union` keeps `["math","code"]` (i.e. `union_numpy_dict` equality assert, `protocol.py:187-198`), then `build_domain_weights(["math","code"], ["math","code","if"])` + `select_routed_teacher_logprobs` returns `[10.0, 20.0]` from teachers `[10,20,30]` — confirms the positional `domain_order → teacher index` contract of `mt_opd.py:55-59`. |
| Fixture `_training_batch` | 22-42 | Encodes the real batch shape: `domain` **and** `extra_info["domain"]` **and** `data_source` all present. Confirms the redundancy noted in §1. |

Coverage gaps in this file, relevant to §8: **no test for an unknown domain string** (the `1.0/n` fallback and the `0.0` loss weight), and **no test for the missing‑`domain` `ValueError`** at `ray_trainer.py:2110-2114`. `tests/test_mt_opd_gradient_share.py:118-129` covers only the *absent‑from‑batch but known* domain; `tests/test_mt_opd_domain_metrics.py` uses arbitrary labels (`"a"`, `"b"`, `"IF"`) but never exercises them through `teacher_domains`.

---

## 10. Things I could not verify / look wrong

1. **`actor_rollout_ref.rollout.reward_mode` has no Hydra schema entry.** It is absent from `verl/trainer/config/rollout/rollout.yaml` (326 lines; I enumerated every top‑level key) and from the `RolloutConfig` dataclass (`verl/workers/config/rollout.py:140-175` — `log_prob_top_k`, `top_k_strategy`, `reward_weight_mode`, `teacher_temperature`, `exopd_lambda` are all there, `reward_mode` is not). Yet `scripts/local/mt_opd.sh:87` and `task_memory/.../60_mt_opd.sh:60` pass it **without** a leading `+`. Under Hydra struct mode that override should fail to compose. `ray_trainer.py:88` reads it defensively (`rollout.get("reward_mode", "opd_kl")`) and `:89-91` deletes it inside `open_dict`, which implies struct *is* on. I found no `OmegaConf.set_struct(..., False)` in `main_ppo.py`. Either an unlisted mechanism supplies the key or the released launcher's override syntax is wrong — I could not settle this without running Hydra.
2. **`mt_opd/domain/{dom}/loss_weight` is sampled, not aggregated** (`ray_trainer.py:2228-2230` uses `dom_w[sel[0]]`). That is exact while `dom_w` is a per‑domain constant, but once `conflict_policy=mask` folds in the per‑sequence `per_seq` factor (`:2218-2222`) the weight varies *within* a domain and the logged number becomes an arbitrary single sequence.
3. **M3‑mask normalization is lost when composed with M1.** `apply_teacher_conflict_policy` renormalizes the token weight to preserve total loss (`mt_opd.py:479-482`), but after `dom_w * per_seq` (`ray_trainer.py:2222`) there is no re‑normalization to token‑weighted mean 1 — so the `M1 + mask` combination does perturb the effective learning rate, unlike M1 alone (`mt_opd.py:333-334`).
4. **Async `domain` propagation is convention‑dependent.** No agent loop I read writes `domain` into `AgentLoopOutput.extra_fields`, which is the only channel that lands in `_postprocess`'s `non_tensor_batch` (`agent_loop.py:795-802`). The test at line 219 constructs that array by hand. Routing works regardless because `batch` retains its own copy, but the test does not prove the production async path emits `domain`.
5. `mt_opd.py:338-402 refresh_opd_advantage` and `apply_teacher_conflict_policy`'s `consensus` branch are reachable but not exercised by anything in `scripts/local/`; whether they are used in the released runs is not determinable from the repo.