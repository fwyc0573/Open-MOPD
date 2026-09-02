MT-OPD file aliases (all absolute):
- `MAIN` = `/data/ycfeng/Open-MOPD/training/verl/verl/trainer/main_ppo.py`
- `RT` = `/data/ycfeng/Open-MOPD/training/verl/verl/trainer/ppo/ray_trainer.py`
- `DPA` = `/data/ycfeng/Open-MOPD/training/verl/verl/workers/actor/dp_actor.py`
- `MT` = `/data/ycfeng/Open-MOPD/training/verl/verl/workers/actor/mt_opd.py`
- `FW` = `/data/ycfeng/Open-MOPD/training/verl/verl/workers/fsdp_workers.py`
- `CA` = `/data/ycfeng/Open-MOPD/training/verl/verl/trainer/ppo/core_algos.py`
- `RWD` = `/data/ycfeng/Open-MOPD/training/verl/verl/trainer/ppo/reward.py`
- `NRM` = `/data/ycfeng/Open-MOPD/training/verl/verl/workers/reward_manager/naive.py`
- `PROTO` = `/data/ycfeng/Open-MOPD/training/verl/verl/protocol.py`
- `CFGR` = `/data/ycfeng/Open-MOPD/training/verl/verl/workers/config/rollout.py`
- `CFGA` = `/data/ycfeng/Open-MOPD/training/verl/verl/workers/config/actor.py`
- `SH` = `/data/ycfeng/Open-MOPD/scripts/local/mt_opd.sh`

Reference config for this trace (from `SH:75-113`): `algorithm.adv_estimator=token_reward_direct`, `actor_rollout_ref.rollout.reward_mode=mt_opd`, `reward_model.enable=True`, `reward_model.model.path=<teacher_0>`, `+mt_opd.teacher_domains=[math,code,if]`, `+mt_opd.n_additional_teachers=2`, `+mt_reward_model_1.*`, `+mt_reward_model_2.*`. Implied defaults: `rollout.log_prob_top_k=256`, `rollout.top_k_strategy=only_stu`, `rollout.reward_weight_mode=student_p` (`CFGR:154-157`).

---

## A. Process entry and construction

1. `MAIN:36-43` — Hydra `@hydra.main(config_path="config", config_name="ppo_trainer")` resolves the full config tree; `main(config)` → `run_ppo(config)`. **In:** CLI overrides. **Out:** OmegaConf `config`. Upstream verl.

2. `MAIN:57-75` — `ray.init` with runtime env from `get_ppo_ray_runtime_env()` merged with `config.ray_kwargs.ray_init.runtime_env`. Upstream.

3. `MAIN:77-97` — `TaskRunner` wrapped as `ray.remote(num_cpus=1)`, `ray.get(runner.run.remote(config))`. Upstream.

4. `MAIN:257` → `MAIN:121-152` — `add_actor_rollout_worker`: `actor.strategy in {fsdp,fsdp2}` and `rollout.mode != "async"` → `role_worker_mapping[Role.ActorRollout] = ray.remote(ActorRolloutRefWorker)` from `FW:161`. Upstream.

5. `MAIN:258` → `MAIN:154-175` — `add_critic_worker` registers `Role.Critic`. Note this registers unconditionally; whether the critic is *used* is decided later by `need_critic(config)` (`RT:778`). For `adv_estimator=token_reward_direct` there is no critic. Upstream.

6. `MAIN:266` → `MAIN:202-226` — `add_reward_model_worker`: `reward_model.enable=True` → `role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)` (`FW:1735`), mapped to `global_pool`. This single registration is reused for **all** MT teachers (see step 15). Upstream mechanism, MT-OPD reuse.

7. `MAIN:269` → `MAIN:228-234` — `add_ref_policy_worker` registers `Role.RefPolicy` only if `algorithm.use_kl_in_reward` or `actor.use_kl_loss`. Default MT-OPD run: neither → no ref worker. Upstream.

8. `MAIN:272-276` — `validate_config(...)`. At this point `actor_rollout_ref.rollout.reward_mode` is still present in the DictConfig (it is only removed in step 12). Upstream.

9. `MAIN:288-303` — tokenizer/processor built; `reward_fn` / `val_reward_fn` = `load_reward_manager(...)` (`RWD:223-279`). With no `reward_model.reward_manager` override, `reward_manager_name="naive"` → `NaiveRewardManager` with `compute_true_reward=True` default (`NRM:83`). Upstream.

10. `MAIN:314-330` — `create_rl_dataset` builds `RLHFDataset` for train/val. The train parquet must carry a per-row `domain` column (consumed at `RT:2109`); MT-OPD hard-requires it.

11. `MAIN:333-346` — `RayPPOTrainer(...)` constructed.

12. `RT:641-642` → `RT:86-92` — **Open-MOPD-specific**: `_pop_direct_opd_rollout_options(config)` reads `actor_rollout_ref.rollout.reward_mode` (default `"opd_kl"`) into `self.direct_opd_options`, then **deletes the key from the rollout config** under `open_dict`. This is required because `RolloutConfig` (`CFGR`) has no `reward_mode` field, so the later `omega_conf_to_dataclass` in the worker would reject it. **Out:** `self.reward_mode="mt_opd"`.

13. `RT:644,657-658` — `self.use_mt_opd = True`; raises if `use_rm` is False (i.e. `reward_model.enable` must be True — the OPD teacher_0 is the reward-model role).

14. `RT:660-777` — **Open-MOPD-specific** MT-OPD config parse. Hydra key paths read: `mt_opd.teacher_domains` (required, list; ordering defines teacher index), `mt_opd.n_additional_teachers`, `mt_opd.domain_weighting` (`domain_routing`|`uniform`), `mt_opd.target_share_domains` + `mt_opd.target_share_values` (parallel lists) **or** `mt_opd.target_gradient_shares` (dict) → `self.mt_target_gradient_shares` (M1), `mt_opd.normalize_reward_scale` (bool or alpha in [0,2], M2), `mt_opd.reward_scale_stat` (`mean|q75|q90|top10`), `mt_opd.reward_scale_direction` (`divide|multiply`), `mt_opd.reward_scale_anchored`, `mt_opd.conflict_policy` (`none|mask|consensus`, M3), `mt_opd.conflict_nats`, `mt_opd.conflict_quantile`. Cross-checks: `RT:696-702` rejects target-share domains not in `teacher_domains`; `RT:740-744` rejects M2 without M1.

15. `RT:1372-1441` → `init_workers`. `RT:1386-1391` registers actor/rollout with `config=self.config.actor_rollout_ref`. `RT:1416` registers `rm_wg` with `config=self.config.reward_model` = **teacher_0**. `RT:1427-1440` **Open-MOPD-specific**: for `i in 1..mt_n_additional`, reads Hydra key `mt_reward_model_{i}`, raises if missing, `OmegaConf.merge(self.config.reward_model, overlay)` and registers colocated class key `mt_rm_{i}`.

16. `RT:1464-1472` — `create_colocated_worker_cls` fuses `{ActorRollout, RewardModel, mt_rm_1, mt_rm_2}` into one worker class per resource pool; `spawn(prefix_set=...)` returns per-prefix worker-group handles. All teachers therefore share the same GPUs as the actor.

17. `RT:1486-1496` — `self.rm_wg.init_model()`, then `self.mt_rm_wgs = [all_wg["mt_rm_1"], all_wg["mt_rm_2"]]` each `init_model()`. `RT:1499-1500` — `self.actor_rollout_wg.init_model()` last (vLLM KV-cache sizing).

18. `MAIN:350` → `RT:1690` — `trainer.fit()`.

---

## B. fit(): per-iteration setup

19. `RT:1702-1707` — `Tracking` logger built from `trainer.project_name` / `experiment_name` / `trainer.logger`.

20. `RT:1712` — `_load_checkpoint()`; `RT:1737-1743` optional `val_before_train`; `RT:1753` `self.global_steps = 1`.

21. `RT:1773-1799` — DAPO dynamic-sampling state (`algorithm.filter_groups.*`). `RT:1787-1789` `_dyn_preserve_domain_balance = (reward_model.reward_manager == "mixed")` — the MT-OPD-relevant domain-quota variant of `_take_complete_groups` (`RT:1809-1853`). Not active for the `SH` recipe (naive manager).

22. `RT:1871` — one dataloader batch. `RT:1881` `batch = DataProto.from_single_dict(batch_dict)`. **tensor_batch:** `input_ids, attention_mask, position_ids`; **non_tensor_batch:** `raw_prompt_ids, domain, data_source, reward_model, extra_info, ...`.

23. `RT:1884-1886` — writes `non_tensor_batch["uid"]` (one uuid per prompt).

24. `RT:1888` → `RT:1145-1165` — `_get_gen_batch`. **Open-MOPD-specific comment/behaviour** (`RT:1146-1151`): the keep-set is `{data_source, reward_model, extra_info, uid, domain}`; everything else in `non_tensor_batch` plus tensors `input_ids/attention_mask/position_ids` is `pop`ped into `gen_batch`. Net effect: `domain` **stays on `batch`** so MT-OPD can route teachers after rollout. **Out:** `gen_batch` (tensor: input_ids, attention_mask, position_ids; non_tensor: raw_prompt_ids…), `batch` (non_tensor only: uid, domain, data_source, reward_model, extra_info).

25. `RT:1891-1894` — `gen_batch.meta_info["global_steps"]`; `gen_batch_output = gen_batch.repeat(rollout.n, interleave=True)`.

26. `RT:1899-1901` → `FW:953-1001` — `actor_rollout_wg.generate_sequences(gen_batch_output)`; vLLM rollout. **Out tensor_batch:** `prompts, responses, input_ids, attention_mask, position_ids` (+ `rollout_log_probs` if `calculate_log_probs`); **meta_info:** `timing` (popped at `RT:1906`).

27. `RT:1940-1941` — `batch = batch.repeat(rollout.n, interleave=True)` (repeats `uid`/`domain` in lockstep with rollouts), then `batch.union(gen_batch_output)`. `PROTO:788-805` `union` merges `batch` / `non_tensor_batch` / `meta_info`, asserting equality on duplicate keys.

28. `RT:1943-1944` → `RT:451-466` — `batch.batch["response_mask"] = attention_mask[:, -response_length:]`, shape `[B, T]`.

29. `RT:1949-1950` → `RT:1654-1684` — `_balance_batch` reorders **rows** of the whole DataProto for DP token balance. `domain` rides along in `non_tensor_batch`, which is why MT-OPD can still route correctly afterwards. Adds `global_seqlen/*` metrics.

30. `RT:1953` — `batch.meta_info["global_token_num"] = attention_mask.sum(-1).tolist()` (consumed by `FW:925-929` for MFU).

---

## C. Student first forward (top-k ids/log-probs)

31. `RT:1957` — gate: `self.use_rm and "rm_scores" not in batch.batch`. True for MT-OPD.

32. `RT:1961` → `FW:1003-1054` — `actor_rollout_wg.compute_log_prob(batch)`. Worker sets `meta_info`: `micro_batch_size=rollout.log_prob_micro_batch_size_per_gpu`, `max_token_len=rollout.log_prob_max_token_len_per_gpu`, `use_dynamic_bsz=rollout.log_prob_use_dynamic_bsz`, `temperature=rollout.temperature`, `top_k=rollout.log_prob_top_k` (=256).

33. `FW:1028` → `DPA:861-942` — `DataParallelPPOActor.compute_log_prob(data, calculate_entropy=True)`. **In tensor_batch (select at `DPA:889`):** `responses, input_ids, attention_mask, position_ids`. Splits to micro-batches (`DPA:891-895`).

34. `DPA:908` → `DPA:241-545` — `_forward_micro_batch(..., top_k=256, student_top_k_ids=None)`. Remove-padding path `DPA:270-322`; forward `DPA:330-337`; `log_probs_all = log_softmax(logits/temperature)` `DPA:362`; sampled-token log-prob `DPA:364-366`; entropy `DPA:375-381`; `_, topk_ids = torch.topk(logits_rmpad, k=256)` `DPA:418`; `topk_log_probs = log_probs_all.gather(-1, topk_ids)` `DPA:421`; Ulysses gather `DPA:424-451`; `pad_input` back and slice `[:, -response_length-1:-1]` `DPA:460-488`. **Returns:** `(entropy[B,T], log_probs[B,T], topk_ids[B,T,K], topk_log_probs[B,T,K])`.

35. `FW:1030-1040` — packs `DataProto`. **Out tensor_batch:** `old_log_probs [B,T]`, `entropys [B,T]`, `student_top_k_ids [B,T,256]`, `student_top_k_log_probs [B,T,256]`, `student_valid_counts [B,T]`; **meta_info:** `temperature`.

36. `RT:1965` — `batch = batch.union(old_log_prob)`; `temperature` enters `batch.meta_info` here and is what `DPA:951` later reads in `update_policy`.

37. `RT:1968-1987` — driver stamps `batch.meta_info`: `global_steps`, `is_plot`, `log_prob_top_k=256`, `top_k_strategy="only_stu"`, `kl_estimator`, `reward_weight_mode="student_p"`, `teacher_temperature`, `reward_mode="mt_opd"`, `top_p_intersec_p`. `RT:1973-1974` **Open-MOPD guard**: `reward_mode in {delta_opd, mt_opd}` requires `log_prob_top_k > 0`.

---

## D. Teacher scoring: teacher_0 then teachers 1..N

38. `RT:1989-1990` → `FW:2662-2871` — `self.rm_wg.compute_rm_score(batch)` = **teacher_0**. Reads `batch.batch["old_log_probs"]`, `student_top_k_ids`, `student_top_k_log_probs`, `response_mask`, `input_ids/attention_mask/position_ids/responses`, and `meta_info[log_prob_top_k / top_k_strategy / teacher_temperature / top_p_intersec_p]`.

39. `FW:2751-2759` → `FW:2023-2399` — teacher `_forward_micro_batch` under `no_grad`. `FW:2356-2357` slices response logits and divides by `teacher_temperature`; `FW:2361` teacher entropy; `FW:2378-2391` → `FW:1911-2015` `_compute_teacher_top_k_log_probs(logits, student_ids, top_k=256, strategy="only_stu")`. For `only_stu` (`FW:1971-1995`): `chunk_log_probs = gather(logits, student_ids) - logsumexp(logits)` → teacher log-prob on the *student's* top-k ids; `valid_counts = K` for all; `overlap_mask` from teacher-topk membership (`FW:1960-1963`, metrics only).

40. `FW:2825-2828` — because `top_k > 0`, **`rm_scores` is deliberately `None`** here; reward computation is deferred to the actor worker.

41. `FW:2839-2862` — **Out tensor_batch:** `teacher_on_student_log_probs [B,T,256]`, `teacher_top_k_ids [B,T,256]`, `teacher_top_k_log_probs [B,T,256]`, `teacher_entropy [B,T]`, `teacher_valid_counts [B,T]`, `overlap_mask [B,T,256]`, `teacher_in_student_mask [B,T,256]`.

42. `RT:1991` — `batch = batch.union(teacher_data)`; these become teacher_0's contribution.

43. `RT:2100-2107` — **Open-MOPD-specific MT branch**. For each `extra_wg` in `self.mt_rm_wgs` (i = 1..N): `extra_wg.compute_rm_score(batch)` (same worker method, different frozen weights), timer `compute_mt_rm_{i}_score`. Only `teacher_on_student_log_probs` is kept, renamed to `mt_teacher_{i}_on_student_log_probs [B,T,256]` and unioned. Every other teacher's `teacher_top_k_*` / `overlap_mask` is discarded.

44. `RT:2109-2114` — reads `batch.non_tensor_batch["domain"]`; raises `ValueError` if absent. This is the routing key.

45. `RT:2116-2126` — assembles `teacher_logps = [teacher_on_student_log_probs, mt_teacher_1_..., mt_teacher_2_...]` by while-loop over `batch.batch` keys; asserts `len(teacher_logps) == len(mt_opd.teacher_domains)`. Index 0 is always `rm_wg` (teacher_0), so `teacher_domains[0]` must name teacher_0's domain.

46. `RT:2128-2133` → `MT:24-62` — `build_domain_weights(domains, domain_order, device, weighting)`. Returns `[B,N]` float32. `domain_routing` → one-hot per row via `{d: i}` dict; a row whose `domain` is not in `domain_order` falls back to `1/N` (`MT:57-61`). `uniform` → all `1/N`.

47. `RT:2134-2136` → `MT:65-87` — `select_routed_teacher_logprobs(teacher_logprobs, domain_weights)`. Stacks to `[N,B,T,K]`, reshapes weights to `[N,B,1,1]`, returns `(w*stacked).sum(0)` → `routed_lp [B,T,K]`. One-hot ⇒ hard selection; uniform ⇒ mean.

48. `RT:2140-2148` → `MT:405-491` — **M3** `apply_teacher_conflict_policy`. Computes per-token top-1 spread `max-min` over teachers (`MT:449-450`), threshold = `conflict_nats` or the `conflict_quantile` of masked spread (`MT:460-468`). `policy="mask"` → returns `(routed_lp, token_weight[B,T] renormalised by denom/kept, metrics)`; `policy="consensus"` → replaces routed target with cross-teacher mean where teachers agree; `"none"` → passthrough with empty metrics. **Metrics out:** `mt_opd/conflict/policy_agree_frac`, `mt_opd/conflict/policy_threshold_nats`, plus `mt_opd/conflict/mask_rescale` or `mt_opd/conflict/consensus_frac`.

49. `RT:2149` — `batch.batch["teacher_on_student_log_probs"] = routed_lp`. **Direct TensorDict assignment, deliberately not `union`** (union asserts tensor equality on duplicate keys, `PROTO:113-119`). Teacher_0's own tensor is overwritten by the routed one; downstream OPD code is unchanged and simply sees "the teacher".

---

## E. OPD reward from the routed teacher

50. `RT:2151-2152` → `FW:1091-1117` — `actor_rollout_wg.compute_distillation_reward(batch)`. Worker adds `meta_info`: `micro_batch_size`, `max_token_len`, `use_dynamic_bsz`, `temperature` (all from `config.rollout.*`).

51. `FW:1106` → `DPA:600-774` — `compute_distillation_reward`. Reads `meta_info[log_prob_top_k, top_k_strategy, kl_estimator, reward_weight_mode, micro_batch_size, temperature, use_dynamic_bsz]`. For `only_stu` the `S_on_T` extra forward at `DPA:622-656` is **skipped** (`strategy not in [only_tch, intersection, union, union-intersection]`) — no additional student forward pass.

52. `DPA:661-663` — reads `student_top_k_ids`, `student_top_k_log_probs` (=`S_logp`), `teacher_on_student_log_probs` (=`T_on_S`, now the routed tensor).

53. `DPA:710-714` — `only_stu` branch: `kl_val = S_logp - T_on_S`; `valid_mask = ones`; `norm_weights = softmax(S_logp)` over K via `compute_reward_weights(..., "student_p")` (`DPA:672-706`, `logsumexp` normalisation + `nan_to_num`); `rm_scores = -kl_val * norm_weights`.

54. `DPA:773-774` — **Out tensor_batch:** `rm_scores [B,T,256]` only (no `union_top_k_*` for `only_stu`). `RT:2153` unions it into `batch`.

55. `RT:2155-2162` — **metrics out:** `mt_opd/rm_scores_mean` = `masked_mean(rm_scores.sum(-1), response_mask)`; `mt_opd/teacher_entropy` = `masked_mean(teacher_entropy, response_mask)`.

56. `RT:2170-2176` → `MT:90-156` — `compute_domain_share_metrics(domains, response_mask, rm_scores)`. Per domain emits `mt_opd/domain/<d>/prompt_share`, `/token_share`, `/prompt_count`, `/mean_response_length`, `/reward_abs_mean` (`|rm_scores|` summed over K, masked, divided by that domain's token count).

57. `RT:2177-2183` → `MT:494-562` — `compute_teacher_conflict_metrics(teacher_logps, response_mask, domains)`. Emits `mt_opd/conflict/spread_mean`, `/contested_frac`, `/spread_max`, plus `mt_opd/conflict/<d>/spread_mean` and `/contested_frac`. Returns `{}` for fewer than 2 teachers.

58. `RT:2189-2201` — M2 anchoring: on the first MT step, if `mt_opd.reward_scale_anchored`, `self._mt_reward_anchor` is captured from the `mt_opd/domain/<d>/reward_abs_mean` values just logged.

59. `RT:2202-2215` → `MT:159-335` — **M1/M2** `compute_domain_loss_weights(...)`. Returns `None` when `target_shares is None` (default → loss untouched). Otherwise: per-domain observed token share (`MT:271`), base scale `target/observed` (`MT:288`); M2 magnitude `mag` per domain from `reward_scale_stat` (mean per-token `|reward|`, or q75/q90/top10 over masked non-zero values, `MT:275-287`); direction `divide` → `sc/mag**alpha` (`MT:321`) or `multiply` → `sc * clamp((mag/ref)**alpha, 0.05, 20)` with `ref` = anchor or running M1-weighted mean (`MT:303-324`); finally token-weighted mean-normalised to 1 (`MT:333-334`). **Out:** `[B]` float tensor.

60. `RT:2218-2222` — M3 `mask` folding: `conflict_token_weight [B,T]` is reduced to a per-sequence mean over `response_mask` and multiplied into `dom_w`, so M1·M2·M3 compose in one channel.

61. `RT:2223-2230` — writes `batch.batch["domain_loss_weight"] = dom_w` (`[B]`) and emits `mt_opd/domain/<d>/loss_weight` (one representative row per domain).

62. `RT:2232-2238` — the non-MT `elif top_k > 0` fallback (plain `opd_kl`) is **not** taken in this trace; MT-OPD's branch at 2100 already called `compute_distillation_reward`.

---

## F. Driver reward assembly and advantage

63. `RT:2241-2337` — SwanLab candidate-count plots guarded by `student_valid_counts`; `RT:2332-2337` pops `student_valid_counts` / `teacher_valid_counts` / `overlap_counts`.

64. `RT:2350-2355` → `RWD:95-112` — `should_reuse_inline_reward(config, use_rm=True, has_rm_scores=True, is_validation=False)` returns `reward_model.enable_validation_reward_async` (False by default) ⇒ **False**. So the inline shortcut is skipped and driver-side scoring runs.

65. `RT:2372-2380` → `RWD:283-301` → `NRM:119-295` — `compute_reward(batch, self.reward_fn)`. With `compute_true_reward=True` the naive manager runs rule scoring as a diagnostic, then `NRM:275-287` returns `reward_tensor = batch.batch["rm_scores"]` (the dense `[B,T,256]` OPD reward) and puts the rule score into `reward_extra_info["true_reward_score"]`. If `reward_kwargs.compute_true_reward=False`, `NRM:126-131` short-circuits before any rule scoring.

66. `RT:2382-2392` — `maybe_apply_rollout_correction` (`algorithm.rollout_correction`); with `old_log_probs` + `entropys` already present, `RT:2396-2398` skips the recompute.

67. `RT:2415-2436` — `metrics["actor/entropy"]` = `agg_loss(entropys, response_mask, loss_agg_mode)`; `metrics["teacher/entropy"]` from `teacher_entropy`; then `entropys` is popped. Only reached when `need_recomputation` is truthy.

68. `RT:2445` — asserts `old_log_probs` in batch. `RT:2447-2454` ref log-prob (skipped, no ref worker). `RT:2457-2460` critic values (skipped).

69. `RT:2473` — `batch.batch["token_level_scores"] = reward_tensor` (`[B,T,256]`). `RT:2474` `_log_train_first_generation`. `RT:2476-2487` `batch.batch["true_reward_score"]` from `reward_extra_infos_dict` (rule score) else the reward tensor itself. `RT:2489-2490` `reward_extra_infos_dict` → `non_tensor_batch`.

70. `RT:2493-2499` — no `use_kl_in_reward` ⇒ `batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]`.

71. `RT:2505-2510` → `RT:1024-1080` — `_log_rollout_data` optionally dumps `<step>.gen_001.jsonl` with per-row `domain` included (`RT:1059`).

72. `RT:2513-2605` — DAPO block, inactive here (`algorithm.filter_groups.enable` unset).

73. `RT:2620-2631` → `RT:469-575` — `compute_advantage(batch, adv_estimator="token_reward_direct", ...)`. Falls to the generic branch `RT:538-561`: `adv_kwargs = {token_level_rewards, response_mask, config}` plus optional `index=uid`, `true_reward_score`, `student_top_k_ids`, `responses`.

74. `RT:561` → `CA:876-902` — `compute_token_reward_direct_advantage`: with 3D rewards it unsqueezes `response_mask` to `[B,T,1]` and returns `advantages = returns = token_level_rewards * response_mask`. `RT:571-572` writes `batch.batch["advantages"]`, `batch.batch["returns"]` (both `[B,T,256]`). `algorithm.clip_advantages` optionally clamps to `[-1,1]` (`RT:573-574`).

75. `RT:2634-3474` — large `val-topk/*` and `val-extrema/*` diagnostic blocks keyed on `overlap_mask` + 3D `advantages` (union-strategy-heavy; most sub-blocks are for `strategy == "union"`), plus SwanLab entropy/advantage plots. All inside `try/except`.

---

## G. Actor update and optimizer step

76. `RT:3476` — `_dump_data_proto(batch, timing_raw, stage="post_advantage")` (`RT:995-1013`), gated by `trainer.data_proto_dump.*`.

77. `RT:3479-3490` — pops `teacher_on_student_log_probs`, `teacher_top_k_ids`, `teacher_top_k_log_probs`, `teacher_entropy`, `overlap_mask`, `teacher_in_student_mask`, `student_log_probs_on_teacher_ids`. Note `mt_teacher_{i}_on_student_log_probs` is **not** popped and stays in the batch through the actor update.

78. `RT:3500-3526` — `batch.meta_info["multi_turn"]`; adaptive KL-loss coefficient path keyed on `actor.adaptive_kl_loss_reward_key` (default `"delta_opd/weighted_reward_mean"`, `RT:3509` / `actor.yaml:89` — a delta-OPD key, absent in MT-OPD metrics, so `RT:3525` sets `actor/adaptive_kl_loss_missing_reward=1` if `use_kl_loss` + `adaptive_kl_loss_coef` are both on). `RT:3526` writes `batch.meta_info["actor_kl_loss_coef"]`.

79. `RT:3527` → `FW:908-951` — `actor_rollout_wg.update_actor(batch)`. Loads FSDP params/optimizer to GPU, enters `ulysses_sharding_manager`, times `update_policy`.

80. `FW:923` → `DPA:944-1225` — `update_policy(data)`. `DPA:949` reads `meta_info["actor_kl_loss_coef"]`; `DPA:951` `meta_info["temperature"]`.

81. `DPA:953-1010` — key selection. Base: `responses, response_mask, input_ids, attention_mask, position_ids, old_log_probs, advantages`. Conditionals: `ref_log_prob` (`use_kl_loss`), `rollout_is_weights`, `format_mask`, **`domain_loss_weight` (`DPA:975-976`, MT-OPD M1)**, `student_top_k_log_probs`, **`teacher_on_student_log_probs` if `opd_refresh_advantage` (`DPA:985-986`, M4)**, `student_top_k_ids`, `union_top_k_ids/log_probs`.

82. `DPA:1014-1016` — `mini_batches = data.split(actor.ppo_mini_batch_size)`; `on_policy = len(mini_batches)==1 and ppo_epochs==1`.

83. `DPA:1021-1030` — per mini-batch: dynamic (`ppo_max_token_len_per_gpu`) or fixed (`ppo_micro_batch_size_per_gpu`) micro-batching; `actor_optimizer.zero_grad()`.

84. `DPA:1036-1038` — per micro-batch reads `response_mask`, `old_log_probs`, `advantages`.

85. `DPA:1071-1083` — `advantages.dim()==3` ⇒ `top_k = advantages.shape[-1]` (256); `student_top_k_ids` from `union_top_k_ids` else `student_top_k_ids`; second **grad-enabled** `_forward_micro_batch(..., top_k=256, student_top_k_ids=...)`. Inside, `DPA:384-413` re-derives packed ids via `indices` and `_align_rmpad_topk_ids_for_ulysses` (`DPA:55-106`) before `log_probs_all.gather` (`DPA:421`). **Out:** `log_prob_for_loss = topk_log_probs [B,T,256]` with graph attached.

86. `DPA:1095-1108` → `MT:338-402` — **M4** `refresh_opd_advantage`, gated on `self._opd_refresh_advantage and not on_policy and "teacher_on_student_log_probs" in model_inputs`. Rebuilds `advantages = -(S_logp.detach() - T_on_S) * softmax(S_logp)` masked by `response_mask`; emits `mt_opd/m4_advantage_refreshed=1.0`. **See finding F2 — the guard is unreachable in this fit() path.**

87. `DPA:1128-1132` — **M1 application**: `advantages = advantages * domain_loss_weight.view(-1,1,1)`. Applied to the advantage, not to `response_mask`, precisely so that `actor/ppo_kl` and `actor/pg_clipfrac` remain unweighted means (`DPA:1116-1127`).

88. `DPA:1140-1160` — `old_log_prob`: on-policy ⇒ `log_prob_for_loss.detach()` (ratio ≡ 1); off-policy ⇒ stored `union_top_k_log_probs` / `student_top_k_log_probs` for the 3D case.

89. `DPA:1176-1188` → `CA:1154-1293` — `get_policy_loss_fn(actor.policy_loss.loss_mode)`; `vanilla` 3D branch `CA:1214-1250`: `ratio = exp(clamp(log_prob - old_log_prob, -20, 20))`; dual-clipped PPO losses with `clip_ratio_low/high`, `clip_ratio_c`; `CA:1240` **sums over the K axis** to get `[B,T]`; `CA:1286` `agg_loss(pg_losses, response_mask, loss_agg_mode)`. **pg_metrics out:** `actor/pg_clipfrac`, `actor/ppo_kl`, `actor/pg_clipfrac_lower`.

90. `DPA:1191-1197` — optional entropy bonus (`actor.entropy_coeff`). `DPA:1199-1209` optional KL loss with `effective_kl_loss_coef` → `actor/kl_loss`, `actor/kl_coef`.

91. `DPA:1211-1219` — `loss = policy_loss * loss_scale_factor` (`1/gradient_accumulation`, or `bsz/ppo_mini_batch_size` for dynamic bsz); `loss.backward()`; `actor/pg_loss` appended.

92. `DPA:1221` → `DPA:839-858` — **`_optimizer_step`**: FSDP/FSDP2 `clip_grad_norm_(max_norm=actor.grad_clip)`; `DTensor.full_tensor()`; **non-finite grad-norm ⇒ `zero_grad()` and skip `optimizer.step()`**, else `actor_optimizer.step()`. Returns `grad_norm`.

93. `DPA:1222-1224` — `actor/grad_norm` appended per mini-batch; `DPA:1224` final `zero_grad()`; returns the `metrics` dict of per-micro-batch lists.

94. `FW:926-936` — worker adds `perf/mfu/actor` (from `meta_info["global_token_num"]`), `perf/max_memory_allocated_gb`, `perf/max_memory_reserved_gb`, `perf/cpu_memory_used_gb`, `actor/lr`; steps the LR scheduler. `FW:939` returns `DataProto(meta_info={"metrics": metrics})`. `FW:944-949` offloads params/optimizer.

---

## H. Metrics emission

95. `RT:3528-3529` — `reduce_metrics(actor_output.meta_info["metrics"])` (mean over the per-micro-batch lists) merged into `metrics`.

96. `RT:3543-3549` — checkpoint on `trainer.save_freq` / last step / ESI (`RT:1512-1558`, writes `<default_local_dir>/global_step_N/actor` + `data.pt` + `latest_checkpointed_iteration.txt`).

97. `RT:3552-3561` — validation on `trainer.test_freq` → `_validate()` metrics merged.

98. `RT:3577-3592` — `training/global_step`, `training/epoch`, then `compute_data_metrics` (`/data/ycfeng/Open-MOPD/training/verl/verl/trainer/ppo/metric_utils.py:80`, with an explicit 3D-advantage branch at lines 159-231 emitting `critic/advantages/mean_sum_over_k`, `/mean_overall`, `/direct_*` etc.), `compute_timing_metrics`, `compute_throughout_metrics`.

99. `RT:3600` — `logger.log(data=metrics, step=self.global_steps)`. The MT-OPD-specific keys emitted this step: `mt_opd/rm_scores_mean`, `mt_opd/teacher_entropy`, `mt_opd/domain/<d>/{prompt_share,token_share,prompt_count,mean_response_length,reward_abs_mean,loss_weight}`, `mt_opd/conflict/{spread_mean,contested_frac,spread_max,policy_agree_frac,policy_threshold_nats,mask_rescale,consensus_frac}`, `mt_opd/conflict/<d>/{spread_mean,contested_frac}`, and timers `compute_mt_rm_{i}_score`, `compute_distillation_reward`.

100. `RT:3602-3603` — `progress_bar.update(1)`; `self.global_steps += 1`; loop back to `RT:1871` for the next iteration.

---

## Open-MOPD-specific vs upstream verl

MT-OPD additions (all traceable above): `RT:76-83` (imports), `RT:86-92` (`reward_mode` pop), `RT:641-777` (config parse), `RT:1427-1440` + `RT:1492-1496` (extra teacher worker groups), `RT:1146-1151` (`domain` retention in `_get_gen_batch`), `RT:2100-2230` (the whole MT branch), all of `MT` (`mt_opd.py`), `DPA:46`/`232-239`/`975-976`/`985-986`/`1095-1108`/`1128-1132`, `CFGA:128-129`, `CFGR:159-162`, `NRM:122-131`. The `token_reward_direct` estimator (`CA:876-902`), 3D policy-loss branch (`CA:1214-1250`), dense top-k teacher scoring (`FW:1911-2015`, `FW:2365-2396`), and `DPA:600-774` are the shared OPD substrate that MT-OPD reuses unchanged.

## Findings worth flagging

**F1 — `reward_mode` is not a declared config field.** `CFGR` has no `reward_mode`; it exists only as a Hydra `+`/plain override that `RT:86-92` deletes before the worker converts `actor_rollout_ref` to `RolloutConfig`. Any code path that constructs the worker config without going through `RayPPOTrainer.__init__` would fail. Same for `mt_opd.*` and `mt_reward_model_{i}.*`, which are `+`-added top-level nodes with no schema.

**F2 — M4 (`opd_refresh_advantage`) is unreachable from `fit()`.** `DPA:985` and `DPA:1098` both require `teacher_on_student_log_probs` in the batch, but `RT:3479-3490` pops that exact key before `update_actor` at `RT:3527`:
```python
keys_to_pop = [
    "teacher_on_student_log_probs",
    ...
]
```
So `select_keys` never picks it up and the refresh branch never fires; `mt_opd/m4_advantage_refreshed` cannot be emitted from this trainer. Additionally `refresh_opd_advantage` is absent from `MT:14-21` `__all__` (imported by name at `DPA:46`, so it still works), and `opd_refresh_advantage` exists in `CFGA:128` but not in `/data/ycfeng/Open-MOPD/training/verl/verl/trainer/config/actor/actor.yaml`. Unit-level coverage exists at `/data/ycfeng/Open-MOPD/training/verl/tests/test_mt_opd_m2_m3.py`; I found no test exercising it through `fit()`.

**F3 — `mt_teacher_{i}_on_student_log_probs` survives into `update_actor`.** `RT:3479-3490` pops teacher_0's tensor but not the renamed extra-teacher tensors. They are not in `DPA:953-1006`'s `select_keys`, so they are dropped by `data.select` at `DPA:1010` — correct result, but they are carried across the driver→worker dispatch of `update_actor` and cost `(N-1) × [B,T,256]` of transfer per step.

**F4 — `teacher_domains[0]` binding is positional and unvalidated.** Index 0 of `teacher_logps` (`RT:2116`) is always `config.reward_model` (teacher_0); indices 1..N come from `mt_reward_model_{i}`. Only the *count* is checked (`RT:2121-2126`). Nothing verifies that `teacher_domains[i]` corresponds to the model actually loaded at that index — a swapped `--teacher` order silently trains every domain against the wrong teacher. `SH:57-60` checks only that the two list lengths match.

**F5 — `adaptive_kl_loss_reward_key` defaults to a delta-OPD key.** `RT:3509` defaults to `"delta_opd/weighted_reward_mean"`, which MT-OPD never produces, so the adaptive-KL path degrades to `actor/adaptive_kl_loss_missing_reward=1` unless the key is overridden.

**F6 — M2 `divide` is documented as measured-harmful but is still the default.** `MT:290-300` and `RT:719-721` both default `reward_scale_direction` to `"divide"`, while the comment records that direction snowballing onto the easiest domain (IF loss weight 27→91, grad token share 57% vs target 33%).