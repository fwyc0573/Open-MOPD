boss YC

# Open-MOPD Architecture Map

*Repo: `/data/ycfeng/Open-MOPD` @ `yc-bench` · vendored verl `0.7.0.dev` (`training/verl/verl/version/version`). All paths absolute. Every claim traceable to file:line.*

---

## 1. Orientation

Open-MOPD (Multi-Teacher On-Policy Distillation) trains one student to absorb three separately-RL'd domain teachers (math / code / instruction-following) by, at every training step, having **all N teachers score the student's own top-K sampled candidate tokens**, routing each prompt to its domain's teacher, and using the resulting per-token reverse-KL as a *dense* `[B, T, K]` advantage instead of a scalar reward. The novelty lives in **`/data/ycfeng/Open-MOPD/training/verl/verl/workers/actor/mt_opd.py`** (562 lines, 7 pure functions, numpy+torch only) plus its single production caller **`ray_trainer.py:2100-2230`**: that block turns `non_tensor_batch["domain"]` into a teacher choice (`build_domain_weights` → `select_routed_teacher_logprobs`), overwrites `batch.batch["teacher_on_student_log_probs"]` with the routed mix so the unmodified single-teacher OPD reward kernel can be reused verbatim (`ray_trainer.py:2149`), and emits one `[B]` tensor `domain_loss_weight` that the actor multiplies into `advantages` (`dp_actor.py:1128-1132`) to force a *requested per-domain gradient share* — the paper's core claim that prompt-level mixing cannot fix, because IF holds ~20% of prompts but ~0.9% of tokens.

---

## 2. Layer map

| Layer | Owning directories | Entry symbol |
|---|---|---|
| **Launchers** (dry-run-by-default shell) | `/data/ycfeng/Open-MOPD/scripts/local/` (`common.sh` 237 L, `sft.sh`, `rl.sh`, `opd.sh`, `mt_opd.sh`, `eval.sh`), root `eval.sh` shim | `local_run_command` (`common.sh:226`) |
| **Trainer driver** (single-controller Ray) | `training/verl/verl/trainer/` — `main_ppo.py` (472 L), `ppo/ray_trainer.py` (3622 L), `ppo/reward.py`, `ppo/validation_sampling.py`, `ppo/kl_controller.py`, `config/**` | `RayPPOTrainer.fit` (`ray_trainer.py:1691`) |
| **Worker layer** (FSDP + vLLM, Ray actors) | `training/verl/verl/workers/` — `fsdp_workers.py` (2908 L), `actor/dp_actor.py` (1225 L), `rollout/`, `engine/fsdp/`, `config/` | `ActorRolloutRefWorker`, `RewardModelWorker` (`fsdp_workers.py:161`, `:1735`) |
| **Algorithm kernels** (pure math) | `workers/actor/mt_opd.py`, `trainer/ppo/core_algos.py` (2083 L) | `compute_domain_loss_weights`, `compute_token_reward_direct_advantage` |
| **Data + reward** | `training/scripts/{sft,rl,math}/`, `experiments/data/`, `verl/utils/dataset/`, `verl/utils/reward_score/`, `verl/workers/reward_manager/` | `build_mix_rl_raw_union.py`, `default_compute_score`, `NaiveRewardManager` |
| **Evals** (standalone, no Hydra) | `/data/ycfeng/Open-MOPD/evals/` (rollout_engine + verifier), `experiments/backend/`, `experiments/analysis/` | `evals.rollout_engine.vllm_rollout`, `evals.verifier.score` |

---

## 3. Module catalog

**★ = Open-MOPD addition · ◐ = upstream file with Open-MOPD patches · ○ = upstream**

### Algorithm kernels

**★ `/data/ycfeng/Open-MOPD/training/verl/verl/workers/actor/mt_opd.py`** — the five mechanisms. No verl imports, no state, no DataProto. `__all__` at `:14-21` lists 6 of 7 functions (`refresh_opd_advantage` omitted, though `dp_actor.py:46` imports it by name).

| Symbol | Line | Signature → returns | Notes |
|---|---|---|---|
| `build_domain_weights` | 24 | `(domains, domain_order, device, weighting="domain_routing") -> [B,N] f32` | one-hot; **unknown label → uniform `1/N` silently** (`:60-61`); `weighting=="uniform"` → all `1/N` (`:51`) |
| `select_routed_teacher_logprobs` | 65 | `(list[[B,T,K]], [B,N]) -> [B,T,K]` | stack→`[N,B,T,K]`, contract with `[N,B,1,1]`. Allocates new tensor (non-aliasing is a pinned invariant) |
| `compute_domain_share_metrics` | 90 | `(domains, response_mask, rm_scores=None) -> dict[str,float]` | 4–5 keys/domain. **3-D `rm_scores` is `sum(dim=-1)` BEFORE `abs()`** (`:134-135`) |
| `compute_domain_loss_weights` | 159 | `(domains, response_mask, target_shares, rm_scores, normalize_reward_scale, reward_scale_stat, reward_scale_direction, reward_anchor) -> [B] f32 \| None` | **M1+M2**. `None` when `target_shares is None` (`:207`) ⇒ bit-identical default. Final rescale to token-weighted mean 1 (`:333-334`) |
| `refresh_opd_advantage` | 338 | `(student_top_k_log_probs, teacher_on_student_log_probs, response_mask, reward_weight_mode) -> [B,T,K]` | **M4. Dead in the released trainer — see §10.** No test anywhere |
| `apply_teacher_conflict_policy` | 405 | `(routed, list[teachers], response_mask, policy, conflict_nats, conflict_quantile) -> (target, token_weight\|None, metrics)` | **M3**. Spread = max−min over teachers at K-index 0. `policy="none"` returns the *same object* (`:440-441`) |
| `compute_teacher_conflict_metrics` | 494 | `(list[teachers], response_mask, domains=None, conflict_nats=1.0) -> dict` | Pure instrumentation. `{}` for <2 teachers |

**◐ `trainer/ppo/core_algos.py`** — `@register_adv_est("token_reward_direct")` at `:876` → `advantages = token_level_rewards * response_mask.unsqueeze(-1)`, `returns = advantages.clone()`; `"token_reward_direct_plus_grpo"` at `:905` (3-tuple return, adds a GRPO outcome term scattered onto the sampled top-K slot via `_scatter_outcome_adv_onto_sampled_topk`, `:970`); `"token_grpo"` at `:352` is **registered but unreachable** (no `**kwargs`, so the forwarded `true_reward_score` TypeErrors). 3-D policy-loss branch at `:1214-1251` (`pg_losses = sum(dim=-1)` at `:1240`).

### Trainer driver

**◐ `trainer/ppo/ray_trainer.py`** (3622 L) — the only production caller of six mt_opd kernels (imports `:76-83`).
- `_pop_direct_opd_rollout_options` `:86-92` — reads `actor_rollout_ref.rollout.reward_mode` (default `"opd_kl"`) then **`del`s it** under `open_dict`; load-bearing because `RolloutConfig` has no such field, so `omega_conf_to_dataclass(self.config.rollout)` (`fsdp_workers.py:610`) would reject it.
- `_ensure_validation_data_source` `:95-131` — per-row backfill of `data_source` from `dataset`; three paths (whole-column when `data_source` absent `:120-123`; per-row when partially null; no-op on shape mismatch).
- `__init__` `:588-797` — 13 ValueErrors; mt_opd parse `:660-777`.
- `init_workers` `:1372-1510` — `mt_rm_{i}` registration `:1427-1440`, `self.mt_rm_wgs` `:1492-1496`.
- `_get_gen_batch` `:1145-1165` — keep-set `{data_source, reward_model, extra_info, uid, domain}` at `:1150`; `domain` **stays on `batch`** and is absent from `gen_batch` in sync mode.
- MT-OPD reward block `:2100-2230`. `compute_advantage` `:469-575` (forwards `true_reward_score`/`student_top_k_ids`/`responses`, unpacks 3-tuple into `data.batch`).
- Per-domain DAPO quota preservation `:1787-1868`, gated on `reward_model.reward_manager == "mixed"`.

**★ `trainer/ppo/kl_controller.py`** (14 L) — `update_kl_loss_coef_from_reward`; multiplicative sign-of-reward controller, mutates `actor_config.kl_loss_coef` in place (`ray_trainer.py:3520`) and ships via `meta_info["actor_kl_loss_coef"]` (`:3526`). Default tracked key is `"delta_opd/weighted_reward_mean"` — inert under mt_opd.

**★ `trainer/ppo/validation_sampling.py`** — `build_validation_repeat_plan` for the heterogeneous val suite (AIME@64 / LCB@10 / IF@1).

### Worker layer

**◐ `workers/fsdp_workers.py`** — `RewardModelWorker` (`:1735`) is repurposed from a scalar RM into a **teacher LM**: `_build_model` `:1801-1885` loads `AutoModelForCausalLM` (docstring at `:1737` still says `AutoModelForTokenClassification` — stale). `_compute_teacher_top_k_log_probs` `:1911-2015` (chunked, 1024 rows) returns teacher log-probs gathered at the *student's* ids + overlap masks. `compute_rm_score` `:2662-2871` returns **no `rm_scores` when `top_k>0`** (`:2825-2828`) — reward math deferred to the actor. `compute_distillation_reward` registered only here (`:1093`); **absent from `megatron_workers.py`**.

**◐ `workers/actor/dp_actor.py`** — `compute_distillation_reward` `:601-774` (only_stu branch `:710-714`: `rm_scores = -(S_logp − T_on_S) * softmax_K(weights)`); `compute_delta_opd_reward` `:777`; `compute_exopd_reward` `:797`; `_forward_micro_batch` `:241-545` (top-K gather, Ulysses re-alignment `:55-106`); `update_policy` `:944-1225` — M1 application at `:1128-1132`, M4 gate at `:1095-1099`.

### Data + reward

| Module | Mark | Role |
|---|---|---|
| `experiments/data/build_mix_rl_raw_union.py` (265 L) | ★ | **The only builder that emits the top-level `domain` column** (`OUTPUT_SCHEMA:57-66`); 13-field unified `extra_info` struct so three domains share one Arrow type |
| `experiments/data/build_mix_rl_union.py` (204 L) | ★ | Restores a legacy 24K selection by greedy subsequence join; no declared schema, pandas passthrough |
| `training/scripts/rl/build_{dapo_math,deepcoder,nemotron_if,livecodebench}_*.py` | ★ | Per-domain 5-column RL parquets — **none emits `domain`** |
| `training/scripts/sft/build_weighted_mix.py` | ★ | MixSFT 7-column schema; released mix 459646/459646/229823 (2:2:1) |
| `training/scripts/sft/prepare_posttrain_tokenizer.py` | ★ | Grafts a post-training chat template onto an unchanged base vocab + writes `generation_config.json` |
| `utils/dataset/domain_weighted_sampler.py` (109 L) | ★ | `DOMAIN_ORDER = ("math","code","if")` hardcoded at `:15`; raises without a top-level `domain` column (`:58-59`); raises on any empty pool (`:66`) |
| `utils/dataset/rl_dataset.py` (507 L) | ◐ | Column-agnostic passthrough (`:338` `row_dict = self.dataframe[item]`) — this is *how* `domain` reaches `non_tensor_batch`. Patched batched-pyarrow/duckdb loader `:37-80` (upstream `load_dataset` fails on nested code testcases) |
| `utils/reward_score/{ttrl_math,rllm_code_reward,instruction_following,if_grm,opd_val_dispatch,mix_rl_dispatch,code_testcase_runners}` | ★ | 8 additions absent from the verl 0.7.0 wheel |
| `workers/reward_manager/naive.py` (295 L) | ◐ | Passes `rm_scores` through verbatim as the training reward; files the rule score under `reward_extra_info["true_reward_score"]` (`:280`) |
| `trainer/ppo/code_reward_pool.py` (570 L) | ★ | Shared subprocess/Ray testcase pool — the actual code-reward engine |

### Evals

`evals/rollout_engine/vllm_rollout.py` (415 L, ★) — multi-process DP, one `CUDA_VISIBLE_DEVICES` slice per rank (`:335-340`). `evals/verifier/score.py` (208 L, ★) — exact-set dataset registry (`:27-31`), reports `official` + `live` + macros. `evals/score_rollouts.py` (244 L, ★) — **rewrites rollout parquets in place** with a `score` column, substring dataset matching. `experiments/backend/param_merge.py` (344 L, ★) — average / task-arithmetic merge baseline. `experiments/analysis/capability_subspace.py` (405 L, ★) — SVD subspace + DACSP gating, **zero callers outside its own test**.

---

## 4. The MT-OPD training step, in order

Config for this trace: `mt_opd.sh:75-119` + `+actor_rollout_ref.rollout.log_prob_top_k=K` (mandatory, see §9).

| # | Call site | What crosses | mt_opd.* key |
|---|---|---|---|
| 1 | `ray_trainer.py:641` → `:86-92` | reads then **deletes** `rollout.reward_mode` | — |
| 2 | `ray_trainer.py:660-777` | 13 `OmegaConf.select` reads → 12 `self.mt_*` attrs; 10 raises at `:666,681,699,714,723,731,735,741,750,762` | **all of them** |
| 3 | `ray_trainer.py:1427-1440`, `:1492-1496` | `OmegaConf.merge(reward_model, mt_reward_model_{i})` → class key `mt_rm_{i}` → `self.mt_rm_wgs` | `n_additional_teachers` |
| 4 | `ray_trainer.py:1881` | `DataProto.from_single_dict` → **`non_tensor_batch["domain"]`** (object array, len B) | — |
| 5 | `ray_trainer.py:1888` → `:1145-1165` | `domain` in keep-set `:1150` ⇒ retained on `batch`, absent from `gen_batch` | — |
| 6 | `ray_trainer.py:1901` | vLLM `generate_sequences` → `prompts/responses/input_ids/attention_mask/position_ids` | — |
| 7 | `ray_trainer.py:1940-1953` | `repeat(n)` + `union`; writes `response_mask [B,T]` `:1943`; `_balance_batch` reorders rows (domain rides along) `:1950` | — |
| 8 | `ray_trainer.py:1961` → `fsdp_workers.py:1030-1035` | +`old_log_probs`, `entropys`, `student_top_k_ids [B,T,K]`, `student_top_k_log_probs [B,T,K]`, `student_valid_counts` | — |
| 9 | `ray_trainer.py:1968-1987` | meta_info: `log_prob_top_k`, `top_k_strategy`, `kl_estimator`(dead), `reward_weight_mode`, `teacher_temperature`, `reward_mode`, `top_p_intersec_p`, `global_steps`, `is_plot`. **`:1973-1974` raises if `top_k<=0`** | — |
| 10 | `ray_trainer.py:1990` → `fsdp_workers.py:2662` | teacher 0 → `teacher_on_student_log_probs [B,T,K]`, `teacher_top_k_ids/log_probs`, `teacher_entropy`, `teacher_valid_counts`, `overlap_mask`, `teacher_in_student_mask`. **No `rm_scores`** | — |
| 11 | `ray_trainer.py:2102-2107` | teachers 1..N-1; keeps only `mt_teacher_{i}_on_student_log_probs` (rename mandatory: `union` asserts equality on duplicate keys, `protocol.py:113-119`) | `n_additional_teachers` |
| 12 | `ray_trainer.py:2109-2126` | reads `domain` (raise if absent); builds `teacher_logps` by while-loop on key presence; arity check vs `teacher_domains` | `teacher_domains` |
| 13 | `ray_trainer.py:2128-2136` → `mt_opd.py:24,65` | `[B,N]` weights → `routed_lp [B,T,K]` | `domain_weighting` |
| 14 | `ray_trainer.py:2140-2148` → `mt_opd.py:405` | **M3** → `(target, conflict_token_weight[B,T]\|None, metrics)` | `conflict_policy`, `conflict_nats`, `conflict_quantile` |
| 15 | **`ray_trainer.py:2149`** | `batch.batch["teacher_on_student_log_probs"] = routed_lp` — direct assignment, overwrites teacher 0. Must precede step 16 | — |
| 16 | `ray_trainer.py:2152` → `dp_actor.py:601-774` | `rm_scores [B,T,K]` | — |
| 17 | `ray_trainer.py:2156-2183` | `mt_opd/rm_scores_mean`, `/teacher_entropy`, `compute_domain_share_metrics`, `compute_teacher_conflict_metrics` (**conflict_nats NOT passed** — §9) | `reward_scale_stat` (indirectly) |
| 18 | `ray_trainer.py:2189-2201` | M2 anchor captured once from `mt_opd/domain/{d}/reward_abs_mean` in the metrics dict just written | `reward_scale_anchored` |
| 19 | `ray_trainer.py:2202-2215` → `mt_opd.py:159` | **M1+M2** → `dom_w [B]` or `None` | `target_gradient_shares` / `target_share_*`, `normalize_reward_scale`, `reward_scale_stat`, `reward_scale_direction` |
| 20 | `ray_trainer.py:2218-2222` | M3 mask collapsed to per-sequence mean and multiplied into `dom_w` (**not re-normalized** — §9) | — |
| 21 | `ray_trainer.py:2224-2230` | **`batch.batch["domain_loss_weight"] = dom_w`**; `mt_opd/domain/{d}/loss_weight` from `dom_w[sel[0]]` | — |
| 22 | `ray_trainer.py:2354` → `reward.py:112` → `naive.py:275-287` | `should_reuse_inline_reward` False (training) ⇒ manager runs, returns `rm_scores` verbatim. **`reward_extra_info` is REBOUND at `naive.py:279`**, discarding the per-row rule metrics | — |
| 23 | `ray_trainer.py:2473,2499` | `token_level_scores` → `token_level_rewards` | — |
| 24 | `ray_trainer.py:2620` → `core_algos.py:895-902` | `advantages [B,T,K] = rewards * mask.unsqueeze(-1)` | — |
| 25 | `ray_trainer.py:3478-3489` | **pops `teacher_on_student_log_probs`** + 6 siblings | — |
| 26 | `ray_trainer.py:3527` → `dp_actor.py:944` | `select_keys` picks `domain_loss_weight` (`:975-976`); M4 gate (`:1095-1099`) **cannot fire** — key popped at step 25 | `opd_refresh_advantage` |
| 27 | `dp_actor.py:1082-1092` | 2nd grad-enabled forward on `student_top_k_ids` → `log_prob_for_loss [B,T,K]` | — |
| 28 | **`dp_actor.py:1128-1132`** | `advantages = advantages * _dw.view(-1, *([1]*(dim-1)))` — rank-generic, dtype-cast. Applied to the *advantage*, not `response_mask`, so `ppo_kl`/`pg_clipfrac` stay unweighted | — |
| 29 | `dp_actor.py:1176-1188` → `core_algos.py:1214-1251` | dual-clip PPO per candidate, `sum(dim=-1)` over K, `agg_loss(token-mean)` | — |
| 30 | `dp_actor.py:1221` → `:839-858` | grad clip; **non-finite grad-norm ⇒ skip `optimizer.step()`** | — |
| 31 | `ray_trainer.py:3600` | `logger.log(metrics, step)` | — |

---

## 5. Data contracts

### SFT parquet (7 flat columns)
`messages: list<struct<role:string, content:string>>`, `domain`, `source`, `dataset`, `split`, `difficulty`, `sample_id` — all `string`. Written zstd as `train-{00000..}.parquet` + `summary.json` (`build_weighted_mix.py:135-145`). Consumed by `MultiTurnSFTDataset` (`multiturn_sft_dataset.py:52`), which reads column names from the **nested** `config["multiturn"]` block (`:63-66`) — the flat `data.messages_key` in `sft_trainer_engine.yaml:25-27` is dead.

### RL prompt parquet (5 columns) + the MT-OPD union (6 columns)
Per-domain builders emit `data_source: string`, `prompt: list<struct<role,content>>`, `ability: string`, `reward_model: struct<style,ground_truth>`, `extra_info: struct<...>`. Per-domain `extra_info` shapes differ (math `{index}`; code `{split,index,source_config}`; IF 7 fields incl. `instruction_id_list`, `instruction_kwargs_json`, `family`).

**MT-OPD adds a 6th top-level `domain: string` column** and normalizes `extra_info` to one 13-field union (`build_mix_rl_raw_union.py:40-66`): `split, index, source_config, sample_id, raw_prompt, instruction_id_list, instruction_kwargs_json, dataset, agent_name, family, domain, source_index, mix_index`. `index` is rewritten to `f"{domain}-{mix_index}"`; `source_index` is `str()`-coerced so math UUIDs and code ints share a type. `row_group_size=192` (interleaved) or 2048 (`--full-union`). Values of `domain` must ⊂ `mt_opd.teacher_domains`; a mismatch does **not** raise (§9).

### Rollout output parquet (eval)
`<output-dir>/{dataset}_rollouts_base{B}[_offset{O}]_rank{R}.parquet` — every input column, plus stamped `temperature/top_p/top_k/max_tokens[/stop_token_ids]`, plus per-completion `completion_index, completion, completion_tokens, finish_reason, stop_reason, generated_special_token_counts`. One row per (input row × completion). **Ragged**: an empty `result.outputs` emits a row lacking the last three keys (`vllm_rollout.py:279-285`).

### Score output JSON
`evals/verifier/score.py`: `{"results":[{dataset, metric, scored_rows, correct, official_total, official, live, official_extra, trunc_rows, repeat_rows, trunc_repeat_rows, trunc_rate, repeat_rate, trunc_repeat_rate}...], "macro_official": float|null, "macro_live": float|null}`. Unscored datasets instead yield `{dataset, scored_rows:0, correct:null, official_total, official:null, live:null, note}` — no `metric`, no behavior keys (ragged).
`evals/score_rollouts.py`: flat `{dataset, rows, metric, correct, scored_rows, total, pct[, truncated, truncation_rate, max_tokens]}` — and **nothing at all** for an unrecognized dataset name.

### Checkpoint layout
SFT (`checkpoint_handler.py:70` + `fsdp_checkpoint_manager.py:172-369`):
```
<default_local_dir>/global_step_N/
  model|optim|extra_state_world_size_{WS}_rank_{R}.pt
  fsdp_config.json                # {"FSDP_version","world_size"} — world_size is read; FSDP_version never is
  huggingface/                    # config.json + tokenizer + generation_config.json — METADATA ONLY, no weights
  data_{dp_rank}.pt
<default_local_dir>/latest_checkpointed_iteration.txt
```
RL/OPD/MT-OPD (`ray_trainer.py:1512-1558`): identical but nested one level deeper under `global_step_N/**actor**/`, plus a single `data.pt` (not per-dp-rank).

---

## 6. Config surface

### `mt_opd.*` — declared in **no** YAML; every key needs `+`

| Key | Default | Legal | Effect | Read at |
|---|---|---|---|---|
| `teacher_domains` | — (**required**) | list[str] | index *i* names teacher *i*; `[0]`=`reward_model`, `[i]`=`mt_reward_model_{i}`. Also the whitelist for target shares | `:662`, raise `:666` |
| `n_additional_teachers` | `0` | int | how many `mt_rm_{i}` groups to spawn; must be `len(teacher_domains)-1` | `:663`, `:1428` |
| `domain_weighting` | `"domain_routing"` | `domain_routing`\|`uniform` | **never validated** — any typo silently selects routing (`mt_opd.py:51`) | `:664` |
| `target_share_domains` + `target_share_values` | `None` | parallel lists | preferred M1 form (a dict-valued Hydra override fails to compose on the executor, comment `:671-673`); length mismatch raises | `:675-688` |
| `target_gradient_shares` | `None` | dict[str,float] | **M1**. `None` ⇒ loss bit-identical. Unnormalized ratios OK. Unknown domain raises | `:690-702` |
| `normalize_reward_scale` | `False` | `bool` \| float ∈[0,2] | **M2** α. Requires `target_gradient_shares` (`:741`). α=0 ≡ M1 | `:706-717` |
| `reward_scale_stat` | `"mean"` | `mean`\|`q75`\|`q90`\|`top10` | **not validated at init** — only inside the kernel (`mt_opd.py:239`), and only when M1 is on, so a typo can be a permanent silent no-op | `:718` |
| `reward_scale_direction` | `"divide"` | `divide`\|`multiply` | `divide` is documented as **measured harmful** (`mt_opd.py:290-296`: IF weight 27→91, grad share 57% vs 33% target) yet is the default | `:719-726` |
| `reward_scale_anchored` | `False` | bool | pins the multiply reference at step 1. Requires `multiply` + `mean` | `:727-738` |
| `conflict_policy` | `"none"` | `none`\|`mask`\|`consensus` | **M3** | `:746-753` |
| `conflict_nats` | `1.0` | float | policy threshold **only** — not the metric threshold (§9) | `:754-756` |
| `conflict_quantile` | `None` | float ∈(0,1) | self-calibrating; overrides `conflict_nats`. `0.0` silently becomes `None` (`:758` truthiness) | `:757-765` |

### Other keys requiring `+` (undeclared anywhere)
`+mt_reward_model_{i}.{enable,model.path,model.input_tokenizer,model.use_remove_padding,model.fsdp_config.param_offload}`, `+teacher_ref_reward_model.*`, `+actor_rollout_ref.rollout.log_prob_top_k`, `+...top_k_strategy`, `+...reward_weight_mode`, `+...teacher_temperature`, `+...top_p_intersec_p`, `+...exopd_lambda`, `+actor_rollout_ref.actor.opd_refresh_advantage`, `+actor_rollout_ref.actor.opd_reward_weight_mode`, `+data.gen_batch_size`, `+data.domain_weights.*`, `+algorithm.filter_groups.domain_weights`, `+trainer.val_only`.

### Declared keys that matter

| Key | Default | Effect |
|---|---|---|
| `actor_rollout_ref.rollout.reward_mode` | `"opd_kl"` (code fallback; **undeclared in YAML and in `RolloutConfig`**) | `opd_kl`\|`delta_opd`\|`mt_opd`\|`exopd`. **No whitelist** — a typo falls to `elif top_k>0` (`:2232`) and trains plain opd_kl silently |
| `actor_rollout_ref.rollout.log_prob_top_k` | **two live values**: `0` at every DictConfig read (`ray_trainer.py:1968`, `fsdp_workers.py:1022,2720`), `256` in the dataclass (`config/rollout.py:154`) | K. `mt_opd` with 0 raises. The dataclass value reaches only the vLLM engine |
| `actor_rollout_ref.actor.loss_agg_mode` | `token-mean` | **M1's entire premise** — under token-mean, gradient share == token share |
| `actor_rollout_ref.actor.kl_loss_coef` | `2.5` (`actor.yaml:85`) — equals `adaptive_kl_loss_max_coef` | the adaptive controller starts pinned at its own ceiling and can only ratchet down |
| `algorithm.adv_estimator` | `gae`; launchers set `token_reward_direct` | 3-D reward passthrough |
| `reward_model.enable` | `False` | the real on/off for the teacher role; every OPD mode raises without it (`:654,658`) |
| `reward_model.model.input_tokenizer` | **`${actor_rollout_ref.model.path}`** (non-null) | non-null ⇒ `_do_switch_chat_template=True` ⇒ decode/re-tokenize path (§9) |
| `data.sampler.class_path` + `class_name` + `data.dataloader_num_workers=0` | `null`/`null`/`8` | all three needed for `DomainWeightedSampler`; the num_workers assert (`main_ppo.py:450`) fires otherwise |

---

## 7. Stage boundaries and artifact handoff

**Three artifact kinds:** HF model dir (loadable), verl FSDP sharded ckpt (**not** loadable), parquet/jsonl (loadable). Everything is local-only — `utils/local_fs.py:13-17` raises on any `"://"`, and upstream `fs.py`/`hdfs_io.py` are gone.

| Boundary | Direct? | Required conversion |
|---|---|---|
| MixSFT → domain RL | **No** | `bash training/scripts/sft/merge_model.sh <ckpt>/global_step_N` → `python -m verl.model_merger merge --backend fsdp`. Reads `<dir>/fsdp_config.json` for `world_size` and hard-codes `<dir>/huggingface` as the config source (`base_model_merger.py:142`); output is bf16-cast unconditionally |
| domain RL → OPD teacher | **No** | merge `global_step_N/**actor**` (one level deeper — passing `global_step_N` fails `merge_model.sh:34`). *Or* pre-empt: `actor_rollout_ref.actor.checkpoint.save_contents=[...,'hf_model']` **is honored here** (`fsdp_workers.py:892`) |
| MixSFT → student | **No** | same merged dir |
| OPD → eval | **No** | merge, then `--model <hf dir>` |
| any stage → resume itself | **Yes** | `trainer.resume_mode=auto` (the default) reads `latest_checkpointed_iteration.txt`. **Reusing one output dir across stages silently resumes instead of starting fresh** |
| merged HF → `param_merge.py` | Yes | reads `model.safetensors.index.json` or a bare `model.safetensors` |
| RL rollout JSONL → RFT SFT parquet | Yes | `cat step*.jsonl \| python -m experiments.data.rft_from_rollouts extract` (stdin only) |

**Blocking defect:** `checkpoint.save_contents=[...,'hf_model']` does **not** work for the SFT stage. `engine/fsdp/transformer_impl.py:159-165` passes the config as `checkpoint_contents=`, which `FSDPCheckpointManager.__init__` swallows into `**kwargs`, leaving `checkpoint_config=None` and the hard-coded defaults in force. Megatron (`engine/megatron/transformer_impl.py:221`) and the PPO actor pass it correctly. So merging after MixSFT is mandatory — and no README mentions it.

---

## 8. Reading order

1. `/data/ycfeng/Open-MOPD/scripts/local/mt_opd.sh` (121 L) — the exact override set; shows what is and is not configured.
2. `/data/ycfeng/Open-MOPD/training/verl/verl/workers/actor/mt_opd.py` (562 L) — the whole novelty, dependency-free, docstrings carry the measured numbers.
3. `ray_trainer.py:86-92, 641-777` — how `reward_mode` and the 13 mt_opd keys are parsed and validated.
4. `ray_trainer.py:2100-2230` — the MT-OPD reward block; every kernel call in production order.
5. `ray_trainer.py:1145-1165` — why `domain` survives rollout (the keep-set).
6. `workers/fsdp_workers.py:1911-2015, 2662-2871` — teacher scoring; why no `rm_scores` when `top_k>0`.
7. `workers/actor/dp_actor.py:601-774` — the OPD reward kernel MT-OPD reuses unchanged.
8. `workers/actor/dp_actor.py:944-1225` — `select_keys`, the 3-D forward, M4 gate, M1 application, `_optimizer_step`.
9. `trainer/ppo/core_algos.py:876-1020, 1214-1251` — `token_reward_direct` and the 3-D policy loss.
10. `ray_trainer.py:1372-1510` — worker-group fan-out; `mt_rm_{i}` registration and colocation.
11. `experiments/data/build_mix_rl_raw_union.py:33-115` — the on-disk `domain` contract.
12. `training/verl/tests/test_mt_opd_gradient_share.py` + `test_mt_opd_m2_m3.py` (179+428 L) — the executable spec for M1/M2/M3, incl. the advantage-scaling ≡ mask-scaling proof at `:132-155`.
13. `training/verl/tests/trainer/ppo/test_mt_opd_batch_routing.py` (237 L) — the domain-plumbing spec; `:58` asserts `"domain" not in gen_batch.non_tensor_batch`.
14. `workers/reward_manager/naive.py:119-295` — how `rm_scores` becomes the advantage and where the rule score goes.
15. `evals/verifier/score.py:59-121` + `evals/score_rollouts.py:168-241` — the two competing scorers and their disagreement.

---

## 9. Traps and sharp edges

**Startup-blocking**

1. **`log_prob_top_k` is not set by any launcher and defaults to 0 on the read path.** `mt_opd.sh:75-119` never sets it; no YAML declares it (`grep` returns nothing under `trainer/config/`); the trainer reads `.get("log_prob_top_k", 0)`. First step raises `reward_mode=mt_opd requires actor_rollout_ref.rollout.log_prob_top_k > 0` (`:1973-1974`). Fix: `--extra '+actor_rollout_ref.rollout.log_prob_top_k=256'`.
2. **`reward_mode` is passed without `+` while being undeclared everywhere.** `mt_opd.sh:87` / `opd.sh:66` emit `actor_rollout_ref.rollout.reward_mode=mt_opd` plainly, while the same scripts correctly use `+` for `mt_opd.*`. `self.config` is struct-mode (`ray_trainer.py:879`), so under standard Hydra semantics this is a composition error, not a silent fallback. Unverified by execution (hydra not installed here). Workaround: `+actor_rollout_ref.rollout.reward_mode=mt_opd`.
3. **Teacher 0 crashes on `raw_prompt`.** `reward_model.model.input_tokenizer` defaults to `${actor_rollout_ref.model.path}` (non-null, `reward_model.yaml:33`) ⇒ `_do_switch_chat_template=True` (`fsdp_workers.py:1810-1813`) ⇒ `_switch_chat_template_token_level` indexes `non_tensor_batch["raw_prompt"]` (`:2513-2519`). But `return_raw_chat` defaults `False` (`legacy_data.yaml:59`), and `_get_gen_batch`'s keep-set excludes `raw_prompt` anyway. `mt_opd.sh:109` sets `input_tokenizer=null` **only for teachers 1..N-1**. Fix: add `reward_model.model.input_tokenizer=null`.
4. **The `domain` column is mandatory and no `training/scripts/rl/` builder emits it.** `ray_trainer.py:2110-2114` raises — *after* all N teacher forwards. The train parquet must come from `experiments/data/build_mix_rl_*_union.py`.
5. **Shipped defaults break the actor and the reward model.** `ActorConfig.__post_init__` asserts one of `ppo_micro_batch_size`/`_per_gpu` (`config/actor.py:143`); `train_batch_size=1` < `ppo_mini_batch_size=256` raises (`:160`); `reward_model.micro_batch_size_per_gpu` is null with `use_dynamic_bsz` false, so `TensorDict.split(None)` (`fsdp_workers.py:2717`). All three need `--extra`.
6. `trainer.save_freq`/`test_freq` = `0` is a `ZeroDivisionError` mid-loop (`sft_trainer.py:323-324` computes the modulo before the `>0` guard). `test_freq>0` without `data.val_files` iterates `None`.

**Silent-wrongness**

7. **An unknown `domain` label does not raise.** `mt_opd.py:60-61` gives it a uniform `1/N` teacher row (distilled from the *average* of all teachers), and if M1 is on, `mt_opd.py:225` assigns it target `0.0` ⇒ **zero gradient for those sequences**, with the remaining domains scaled up. Only trace: an extra `mt_opd/domain/<label>/*` metric series.
8. **Teacher↔domain binding is positional and unverified.** Only the *count* is checked (`:2121-2126`). Swapping two `--teacher` paths routes math prompts to the code teacher, silently. `mt_opd.sh:58-59` checks list lengths only.
9. **`mt_opd/conflict/contested_frac` is pinned at 1.0 nats regardless of `conflict_nats`.** `ray_trainer.py:2177-2183` passes only `teacher_logprobs/response_mask/domains`; the kernel default (`mt_opd.py:498`) applies. Policy and metric thresholds diverge whenever you tune `conflict_nats`.
10. **M1 + M3-mask silently changes the loss scale.** Each factor alone has token-weighted mean 1; their product at `ray_trainer.py:2222` is not re-normalized. Both docstrings advertise LR preservation.
11. **M3 "mask" does not mask tokens.** The `[B,T]` weight is collapsed to a per-sequence mean (`:2218-2221`); contested tokens inside a kept sequence contribute full gradient. Unit tests verify the per-token contract the integration never exercises.
12. **M2 `divide` is the default and is documented as measured-harmful** (`mt_opd.py:290-296`). It also has no upper clamp (`multiply` clamps to `[0.05,20]`), so `mag=1e-6, α=1` multiplies the scale by 1e6.
13. **Anchored M2 can abort mid-run.** The anchor is captured from the first MT batch only and filters to domains present then (`:2189-2201`); a later batch containing a missing domain raises at `mt_opd.py:245-248`. The quota machinery that would guarantee presence is gated on `reward_manager=="mixed"`.
14. **`torch.quantile` hard-fails above 2²⁴ elements** (verified on the installed torch 2.5.1). Two reachable sites: `mt_opd.py:468` (global-batch spread, whenever `conflict_quantile` is set) and `:287` (`q75`/`q90`). At the docstrings' own 11570-token math responses, ~1450 math sequences cross it. `top10` uses `topk` and is safe.
15. **`only_stu` is the only safe strategy under MT-OPD.** Routing replaces *only* `teacher_on_student_log_probs` (`:2149`); `teacher_top_k_ids`/`overlap_mask` still belong to teacher 0. And `top_k_strategy=top_p_intersec` is declared in `config/rollout.py:155` but has no branch in `compute_distillation_reward` ⇒ `UnboundLocalError` at `dp_actor.py:772`. `only_tch` matches shapes but resolves the outcome scatter against the wrong id axis.
16. `actor/pg_clipfrac` is inflated by up to K in the 3-D branch: `masked_mean(values_(B,T,K), mask_(B,T,1))` sums B·T·K clipped entries over a B·T denominator (`core_algos.py:1249`, `torch_functional.py:163-185`). `pg_clipfrac_lower` is hard-coded `0.0` (`:1251`).
17. `algorithm.use_kl_in_reward=True` broadcast-fails against a 3-D `token_level_scores` (`ray_trainer.py:437`). Nothing validates the combination.
18. `mt_opd/domain/{d}/loss_weight` logs one representative sample (`dom_w[sel[0]]`) — exact under M1/M2, arbitrary once M3-mask varies weights within a domain.
19. `evals/score_rollouts.py` **overwrites your rollout parquets in place** (`:224-226`), and its per-row code graders swallow every exception to `0.0` (`:106,140`) — a missing LCB checkout stamps `score=0.0` everywhere with exit 0.
20. `_get_gen_batch` `domain` retention differs by mode: `"domain" not in gen_batch` in sync (asserted by test), present in async via the `:1162` blanket update. The async agent loops do not put `domain` in `extra_fields`, so the surviving copy is the one on `batch`.
21. Environment: `pip install --user` throughout `install_requirements.sh`; **no vllm, no flash-attn, no torch installed** by it, while all three PPO launchers hard-code `rollout.name=vllm` and `mt_opd.sh:110` sets `use_remove_padding=True` (needs flash-attn). `setup.py` pins `numpy<2.0.0`, so installing verl downgrades a numpy-2 host.
22. In dry-run mode the launchers validate almost nothing (paths, GPUS, NODES all `--run`-gated); only the `TRAINING_DIR` existence check (`common.sh:170`) and the mt_opd arity check fire.

---

## 10. Gaps in the release

**Half-wired**

1. **M4 (`opd_refresh_advantage`) is unreachable through `RayPPOTrainer.fit`.** `dp_actor.py:985` and `:1098` both require `teacher_on_student_log_probs` in the batch, but `ray_trainer.py:3479` pops exactly that key and `update_actor` is at `:3527` — 38 lines later. Verified by reading both. So `mt_opd/m4_advantage_refreshed` can never be emitted. Also: `refresh_opd_advantage` is absent from `mt_opd.py`'s `__all__`, `opd_refresh_advantage` exists in `config/actor.py:128` but in no YAML, and the function has **zero tests**.
2. **`self.mt_reward_scale_stat` is missing from the non-mt_opd else-branch** (`ray_trainer.py:766-777` sets 11 of 12 siblings). Latent `AttributeError`; harmless only because the sole read (`:2208`) is inside the mt_opd branch.
3. **Megatron cannot run any OPD mode, and fails loudly mid-step.** `compute_distillation_reward` is registered only on the FSDP worker (`fsdp_workers.py:1093`); `megatron_workers.py` has only `compute_rm_score`. `ray_trainer.py:2152` raises — after routing, M3, and the teacher-key overwrite have already executed.
4. **`mt_teacher_{i}_on_student_log_probs` survives into `update_actor`.** Not in the pop list, not in `select_keys` — dropped by `data.select`, but transported across the driver→worker dispatch at `(N-1) × [B,T,K]` per step.
5. `token_grpo` (`core_algos.py:352-450`) is registered and unreachable: no `**kwargs`, so the always-forwarded `true_reward_score` TypeErrors. Same defect breaks upstream `grpo_vectorized`.
6. `DomainWeightedSampler` has **zero production callers** — `ray_trainer.py:1797` imports only `weighted_quotas` from that module.
7. `experiments/analysis/capability_subspace.py` (405 L, 68 tests) has zero callers; its own test suite documents a negative result (`test_capability_subspace.py:454-465`: "when the delta is high-rank, a fixed small budget retains little either way, so the method has no signal to exploit"). `orthonormal_basis` (`:140`) uses unpivoted `torch.linalg.qr`, so the docstring's "rank-revealing" claim is not met.
8. `algorithm.mask_truncated_samples` / `overlong_filter` / the `norm_adv_by_std_in_grpo` argument are inert for the OPD estimators (forwarded only inside the hard-coded GRPO branch, `ray_trainer.py:533`). `clip_advantages` clamps `advantages` but not `returns`.
9. `NaiveRewardManager.compute_true_reward` (`naive.py:83`) is a constructor-only flag with no Hydra key and no caller ⇒ the `:126-131` fast path is unreachable; the expensive rule pass always runs, then its metrics are discarded by the rebind at `:279`.
10. `actor_rollout_ref.rollout.kl_estimator` is fully dead: read at `ray_trainer.py:1970`, assigned to an unused local at `dp_actor.py:614`, and undeclared, so `+`-appending it would break the `RolloutConfig` conversion.
11. `fsdp_config.json`'s `FSDP_version` is written and never read. `evals/verifier/score.py` re-exports 7 unused symbols purely as a test surface. `build_deepcoder_rl_dataset.py:150-151` has an identical-branch ternary; `:95-97` is a tautological assert.

**Documented but absent** (all verified missing, and absent from `git ls-files`)

- `evals/rollout_engine/build_data.py` (`evals/README.md:31-34`) — **there is no eval-input builder anywhere in the repo**; the eval parquet schema is only inferable from its readers.
- `evals/misc/fetch_livecodebench_assets` (README:76), `evals/rollout_engine/scripts/run_all.sh` (README:140-143), `evals/verifier/third_party/{repos.lock.json,patches/,repos/}` (README:92-96 — so all four `manage_third_party` commands raise `FileNotFoundError` at `:103`, even `--help`), and the entire `evals/data/` tree (so `--data-dir` silently falls back to the hardcoded `DATASET_TOTALS` table and `--lcb-subsets` becomes a no-op).
- The 24K selection template `build_mix_rl_union.py` restores from is produced by nothing in-repo and is not among the released HF artifacts ⇒ that CLI is unrunnable.
- `experiments/backend/README.md` documents `--base/--model/--output`; the real flags are `--mode/--input NAME=PATH/--reference/--output-dir`.
- `merge_model.sh` — the mandatory inter-stage conversion — is mentioned in neither `README.md` nor `scripts/local/README.md`. Its own usage text is the only documentation.
- Numeric doc drift: `evals/README.md:116-118` claims `max_num_seqs=2048 / max_num_batched_tokens=8192`; the code is 1024/None. `evals/verifier/README.md:13` claims `score.py` is the only scoring entrypoint; `evals/score_rollouts.py` ships.

**Not recoverable from the repo**

No YAML or script anywhere sets a single M1/M2/M3/M4 key, `mt_opd.domain_weighting`, `reward_model.reward_manager`, `data.sampler.*`, `data.domain_weights`, or `custom_reward_function.path=.../opd_val_dispatch.py`. `mt_opd.sh` as shipped therefore produces **naive MT-OPD** (routing only): `target_gradient_shares=None` ⇒ `compute_domain_loss_weights` returns `None`, `conflict_policy="none"`. The settings behind the paper's headline numbers, and the `divide`-vs-`multiply` / anchored / `mean`-vs-`q90` / `none`-vs-`mask` choices, are not in this release. The measured constants quoted throughout the docstrings (response lengths 11570/9032/364; spread mean 0.131 nats, max 29.2, 0.70% >1.0; IF weight 27→91; per-token |reward| vs length r=−0.87/−0.80) come from run logs not present here.