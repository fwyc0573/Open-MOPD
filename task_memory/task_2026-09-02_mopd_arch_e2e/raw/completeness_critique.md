**Verified against source. Branch is `yc-mopd`, not `yc-bench` (map header).**

## (1) Missing entirely

**a. `verl/experimental/reward/reward_loop/` — 1296 L, a second registry-based reward subsystem.** `@register("mixed")` at `experimental/reward/reward_loop/mixed.py:44` (164 L) routes each completed trajectory by `infer_domain(...)` at `:141`, imported from `utils/reward_score/mix_rl_dispatch.py:27` — **a second, independent domain-classification source of truth** that never touches `non_tensor_batch["domain"]`. `opd_validation.py:132` (231 L) is the actual consumer of the `opd_val_dispatch` file the map lists as an inert reward_score module. Also uncataloged: `workers/reward_manager/mixed.py` (58 L), even though the map cites `reward_manager=="mixed"` twice (§3, trap 13).

**b. The reference policy and the KL-loss term.** `Role.RefPolicy` registration `ray_trainer.py:1404-1410`, wg bind `:1479`, `compute_ref_log_prob` in the fit loop `:2446-2453`, `ref_log_prob` in `select_keys` `dp_actor.py:963`, and the term itself `dp_actor.py:1199-1209`. The 31-step trace has no ref step; §3 catalogs `kl_controller.py` and §6 highlights `kl_loss_coef=2.5` but the map never says what that coefficient multiplies.

## (2) Overconfident

**§1's headline.** "`domain_loss_weight` … the actor multiplies into `advantages` … to force a *requested per-domain gradient share*." The source applies `_dw` **only to `advantages`** (`dp_actor.py:1128-1132`), i.e. only to `pg_loss`. The entropy bonus (`:1194-1197`) and `kl_loss * effective_kl_loss_coef` (`:1207`) are aggregated over the raw `response_mask` and remain domain-uniform. With `use_kl_loss=true` at the map's own `kl_loss_coef=2.5`, the achieved share is not the target. M1 rebalances one of three loss terms — the map states it as rebalancing the gradient.

**Step 26 / §10.1 on M4.** The gate has a third condition the map omits: `and not on_policy` (`dp_actor.py:1097`), where `on_policy = len(mini_batches)==1 and ppo_epochs==1` (`:1016`; `actor/actor.yaml:98` ships `ppo_epochs=1`). M4 is dead for two independent reasons; "Verified by reading both" understates the analysis.

**Trap 13's implied escape** (set `reward_manager=mixed` to guarantee domain presence for anchored M2). `MixedRewardManager.__call__` raises unless async rollout already materialized `rm_scores` (`workers/reward_manager/mixed.py:47-51`), and it never files `true_reward_score` (contrast `naive.py:280`) which `compute_advantage` forwards. Not a config flip.

**Trap 15 / §6's strategy enumeration.** `compute_distillation_reward` has a sixth branch, `union-intersection` (`dp_actor.py:750`), absent from `RolloutConfig`'s declared list (`workers/config/rollout.py:155`).

## (3) Wrong citations

- Trap 15: `UnboundLocalError` at `dp_actor.py:772` — line 772 is blank. The unbound read is `res_tensors["rm_scores"] = rm_scores` at **`:773`**.
- Trap 16: `pg_clipfrac` at `core_algos.py:1249` — actual **`:1250`**.
- §3: 3-D `rm_scores` "sum BEFORE abs" at `mt_opd.py:134-135` — the sum is `:134`, `.abs()` is `:137`; 135-136 are a comment.
- Step 7: "writes `response_mask` `:1943`" — 1943 is the `if "response_mask" not in …` guard; the write is 1944.
- Step 25: `:3478-3489` — the list literal is 3479-3487, pop loop 3488-3490 (§10.1's own `:3479` is right).

Everything else I sampled checks out: all seven `mt_opd.py` defs (24/65/90/159/338/405/494), `ray_trainer.py` 86, 95, 588, 660, all ten raises (666,681,699,714,723,731,735,741,750,762), 1145/1150/1162, 1968/1970/1973-74, 2100-2232, 3479, 3527, `dp_actor.py` 46/234/601-774/975-976/985/1016, `fsdp_workers.py` 161/610/892/1030-1035/1093/1735/1737/1801/1810-1813/1911/2513/2662/2717/2825-2828, `naive.py:279-280`, `core_algos.py` 352/876/970/1240/1251, `domain_weighted_sampler.py` 15/58-59/66, torch 2.5.1.

## (4) What a reader still cannot do

- **Know where in the loss M1 lands** (see §2a) — and that `agg_loss(token-mean)`'s denominator is `response_mask.sum()`, unchanged by `_dw`, which is *why* the token-weighted-mean-1 rescale (`mt_opd.py:333-334`) preserves the LR. Without this, a reader adding M5 will attach it to the wrong tensor.
- **How a per-domain quantity reaches the worker.** `domains` is a numpy object array and is never dispatched; `update_policy` has no domain label. Any new per-domain signal must be materialized as a `[B]`-aligned tensor into `batch.batch` inside `ray_trainer.py:2202-2230`, because `data.split(ppo_mini_batch_size)` (`dp_actor.py:1014`) and `prepare_dynamic_batch` (`:1023`) reorder and reshard rows. The map states the *symptom* (a comment at `:2199`) but not the rule.
- **`masked_mean` semantics for any new 3-D metric**: `s / (mask.sum(axis) + 1e-8)` with no shape check (`utils/torch_functional.py:171-185`), so a `[B,T,1]` mask over `[B,T,K]` values silently divides by a K-times-too-small denominator. This is the general trap behind trap 16, not a one-off.
- **The teacher-addition contract**: `mt_teacher_{i}_on_student_log_probs` must be a *fresh* key because `union_tensor_dict` asserts `.equal()` on duplicates (`protocol.py:117-119`), and `mt_rm_{i}` classes are built by `OmegaConf.merge(reward_model, mt_reward_model_{i})` (`ray_trainer.py:1427-1440`) — the map lists both facts but not that they are the two things you must satisfy to add teacher N+1.
- **Which of the two domain-classification paths governs**, and that nothing validates `domain` values against `mt_opd.teacher_domains` (trap 7 covers the silent-uniform fallback but not the `infer_domain` divergence).