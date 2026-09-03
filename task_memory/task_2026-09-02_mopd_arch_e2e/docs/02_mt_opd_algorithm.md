# The MT-OPD Algorithm — Code-Level Walkthrough

## Modification History

| Date       | Summary of Changes                                                    |
| ---------- | --------------------------------------------------------------------- |
| 2026-09-02 | First version. Kernel (`mt_opd.py`, 563 lines) and the trainer reward block (`ray_trainer.py:2100-2230`) read line-by-line; all claims re-verified against source. |
| 2026-09-04 | Updated launcher, teacher-binding, and conflict-threshold behavior after double-check. |

**Prerequisite.** [`01_architecture.md`](01_architecture.md) for the layer map and data
contracts. This document is the algorithm.

---

## 1. The problem the algorithm solves

Take three teachers, each RL-trained on one domain, and one student. Distill all three into
the student on-policy. The obvious recipe — route each prompt to its domain's teacher and
sum the losses — is what the paper calls **Naive M-OPD**, and it lands 3.50 points below the
domain-routed RouteOPD reference.

The diagnosis, written into the code as comments with the measured numbers:

> Student response lengths are math **11570** / code **9032** / IF **364** tokens. Under
> `loss_agg_mode=token-mean` every token contributes equally to the loss, so a domain's
> gradient share is its **token** share — which made IF hold 20% of prompts but 0.9% of the
> gradient, and made the 2:2:1 → 3:3:1 prompt-mixture sweep move token share by only
> 0.1–0.3pp.
> — `verl/workers/actor/mt_opd.py:171-175`

So the mixture ratio you *think* you are setting (prompts) is not the quantity that
determines learning (tokens). Reaching an equal gradient share by sampling alone would need
~32× more IF prompts than math, which does not fit in a batch
(`mt_opd.py:177-178`). Scaling the **loss** reaches any target share without touching
sampling: weight domain `d` by `target_d / observed_d`.

That is M1. Three more mechanisms sit on top:

| ID | Name | What it corrects | Kernel |
| --- | --- | --- | --- |
| **M1** | target gradient shares | token share ≠ prompt share | `compute_domain_loss_weights` (`:159`) |
| **M2** | reward-scale normalization | gradient share is really `token_share × reward_magnitude` | same function, `alpha > 0` path (`:231-326`) |
| **M3** | teacher-conflict policy | the routed teacher's target is not a consensus on contested tokens | `apply_teacher_conflict_policy` (`:405`) |
| **M4** | advantage refresh | the stored reward describes a student that no longer exists after an inner update | `refresh_opd_advantage` (`:338`) — **unreachable, see §7** |

---

## 2. One MT-OPD step, in execution order

All line numbers from `training/verl/verl/trainer/ppo/ray_trainer.py` unless noted.

### Phase A — setup, once

| # | Where | What |
| --- | --- | --- |
| A1 | `:641` → `:86-92` | `_pop_direct_opd_rollout_options` reads `actor_rollout_ref.rollout.reward_mode` (fallback `"opd_kl"`), then **deletes** the key under `open_dict`. The delete is load-bearing: `RolloutConfig` has no such field, so the worker-side dataclass conversion would reject it. |
| A2 | `:657-658` | raises when an OPD mode is set without `reward_model.enable=True` — the teacher role is what makes distillation possible at all |
| A3 | `:660-765` | 13 `OmegaConf.select` reads → 12 `self.mt_*` attributes, 10 `ValueError` sites. Full table in §4 |
| A4 | `:1427-1440` | for `i` in `1..n_additional_teachers`: `OmegaConf.merge(reward_model, mt_reward_model_{i})`, registered under the class key `mt_rm_{i}`. **Raises if the config key is missing.** |
| A5 | `:1492-1496` | each `mt_rm_{i}` worker group is `init_model()`-ed and appended to `self.mt_rm_wgs` |

With `reward_model.enable_resource_pool=False` (the default) everything colocates on
`global_pool = [n_gpus_per_node] * nnodes` (`main_ppo.py:181-199`), so a student, a
reference, and N teachers share the same GPUs.

### Phase B — per step, before the MT block

| # | Where | Adds to the batch |
| --- | --- | --- |
| B1 | `:1881` | `DataProto.from_single_dict` → **`non_tensor_batch["domain"]`** as an object array of length B |
| B2 | `:1888` → `:1145-1165` | `_get_gen_batch`; the keep-set at `:1149-1151` includes `"domain"`, so domain stays on `batch` and is **absent from `gen_batch`** in sync mode (asserted by `tests/trainer/ppo/test_mt_opd_batch_routing.py:58`) |
| B3 | `:1901` | vLLM generation → `prompts`, `responses`, `input_ids`, `attention_mask`, `position_ids` |
| B4 | `:1940-1953` | `repeat(n)` + `union`; writes `response_mask [B,T]` at `:1943`; `_balance_batch` reorders rows and domain rides along |
| B5 | `:1961` → `fsdp_workers.py:1030-1035` | actor forward → `old_log_probs`, `entropys`, **`student_top_k_ids [B,T,K]`**, **`student_top_k_log_probs [B,T,K]`**, `student_valid_counts` |
| B6 | `:1968-1987` | meta_info assembly. **`:1973-1974` raises** when `reward_mode ∈ {delta_opd, mt_opd}` and `log_prob_top_k <= 0` |
| B7 | `:1990` → `fsdp_workers.py:2664` | **teacher 0** → `teacher_on_student_log_probs [B,T,K]`, `teacher_top_k_ids/log_probs`, `teacher_entropy`, `teacher_valid_counts`, `overlap_mask`, `teacher_in_student_mask`. **No `rm_scores`** — the worker returns none when `top_k > 0` (`fsdp_workers.py:2825-2828`), because only the actor holds the student's log-probs |

### Phase C — the MT block, `:2100-2230`

| # | Line | What crosses | Governed by |
| --- | --- | --- | --- |
| C1 | `2102-2107` | loop `self.mt_rm_wgs`: each extra teacher's `compute_rm_score(batch)`, then `union` under the renamed key `mt_teacher_{i}_on_student_log_probs`. The rename is mandatory — `DataProto.union` asserts equality on a duplicate key (`protocol.py:113-119`) | `n_additional_teachers` |
| C2 | `2109-2114` | `domains = batch.non_tensor_batch.get("domain", None)`; **raises** if `None`. This fires *after* every teacher forward has already run | — |
| C3 | `2116-2126` | builds `teacher_logps` by a while-loop on key presence, then checks `len(teacher_logps) == len(self.mt_teacher_domains)`. **Only the count is checked** | `teacher_domains` |
| C4 | `2128-2136` | `build_domain_weights` → `[B,N]`, then `select_routed_teacher_logprobs` → `routed_lp [B,T,K]` | `domain_weighting` |
| C5 | `2140-2148` | **M3** `apply_teacher_conflict_policy` → `(routed_lp, conflict_token_weight [B,T] or None, metrics)`; `metrics.update(_cm)` at `2148` | `conflict_policy`, `conflict_nats`, `conflict_quantile` |
| C6 | **`2149`** | `batch.batch["teacher_on_student_log_probs"] = routed_lp` — a **direct assignment that overwrites teacher 0's tensor**. This is the whole trick: it lets the unmodified single-teacher OPD reward kernel be reused verbatim | — |
| C7 | `2151-2153` | `self.actor_rollout_wg.compute_distillation_reward(batch)` → `rm_scores [B,T,K]` | — |
| C8 | `2155-2161` | `mt_opd/rm_scores_mean` (from `rm_scores.sum(dim=-1)`), `mt_opd/teacher_entropy` | — |
| C9 | `2169-2183` | `compute_domain_share_metrics` (the diagnosis) and `compute_teacher_conflict_metrics` (free, reuses the discarded teachers) | — |
| C10 | `2189-2201` | M2 anchor, captured **once**, from the `mt_opd/domain/{d}/reward_abs_mean` values just written into `metrics` | `reward_scale_anchored` |
| C11 | `2202-2215` | **M1+M2** `compute_domain_loss_weights` → `dom_w [B]` or `None` | `target_gradient_shares` / `target_share_*`, `normalize_reward_scale`, `reward_scale_stat`, `reward_scale_direction` |
| C12 | `2218-2222` | M3's `[B,T]` token weight is collapsed to a per-sequence mean and multiplied into `dom_w` | — |
| C13 | `2224-2230` | **`batch.batch["domain_loss_weight"] = dom_w`**, and one representative `mt_opd/domain/{dom}/loss_weight` per domain from `dom_w[sel[0]]` | — |

### Phase D — reward manager, advantage, update

| # | Where | What |
| --- | --- | --- |
| D1 | `:2354` → `ppo/reward.py:112` → `naive.py:275-287` | the reward manager runs and returns `rm_scores` **verbatim** as the training reward; the rule-based score is filed under `reward_extra_info["true_reward_score"]`. Note `reward_extra_info` is rebound at `naive.py:279`, discarding the per-row rule metrics |
| D2 | `:2473`, `:2499` | `token_level_scores` → `token_level_rewards` |
| D3 | `:2620` → `core_algos.py:895-902` | `advantages = token_level_rewards * response_mask`, with `.unsqueeze(-1)` on the mask only when the reward is 3-D; `returns = advantages.clone()`; whole block under `torch.no_grad()` |
| D4 | `:3478-3489` | **pops `teacher_on_student_log_probs` and 6 siblings** |
| D5 | `:3527` → `dp_actor.py:944` | `update_actor`; `select_keys` picks up `domain_loss_weight` (`dp_actor.py:975-976`) |
| D6 | `dp_actor.py:1082-1092` | second, gradient-enabled forward on `student_top_k_ids` → `log_prob_for_loss [B,T,K]` |
| D7 | **`dp_actor.py:1128-1132`** | `advantages = advantages * dw.view(-1, *([1]*(dim-1)))` — **M1 lands on the advantage, not on `response_mask`**, so `ppo_kl` and `pg_clipfrac` stay unweighted |
| D8 | `dp_actor.py:1176-1188` → `core_algos.py:1214-1251` | dual-clip PPO per candidate, `sum(dim=-1)` over K, `agg_loss(token-mean)` |
| D9 | `dp_actor.py:1221` → `:839-858` | grad clip; **a non-finite grad norm skips `optimizer.step()`** |
| D10 | `:3600` | `logger.log(metrics, step)` |

---

## 3. The five kernels, with their actual math

### 3.1 `build_domain_weights` — `mt_opd.py:24-62`

```
n = len(domain_order)
if domains is None or weighting == "uniform":  return full((B, n), 1/n)
for each sample i:
    j = domain_to_idx.get(str(domains[i]), -1)
    weights[i, j] = 1.0        if j >= 0
    weights[i, :] = 1/n        otherwise          # <-- :60-61
```

**The silent path.** A domain label not in `teacher_domains` does **not** raise. It gets a
uniform row, meaning that sample is distilled from the *average* of all teachers. Combined
with M1 (§3.4) it then gets target share `0.0` and therefore **zero gradient**. The only
trace is an extra `mt_opd/domain/<label>/*` metric series.

### 3.2 `select_routed_teacher_logprobs` — `mt_opd.py:65-87`

```
stacked = stack(teacher_logprobs, dim=0)              # [N, B, T, K]
w       = domain_weights.transpose(0,1).reshape(n,-1,1,1)   # [N, B, 1, 1]
return (w * stacked).sum(dim=0)                       # [B, T, K]
```

Raises on a teacher-count mismatch (`:75-79`) and on a non-3-D tensor (`:81-82`). Returns a
new tensor, never an alias — a property `test_mt_opd_m2_m3.py` pins.

### 3.3 `compute_domain_share_metrics` — `mt_opd.py:90-156`

Per domain, 4 keys always plus 1 when `rm_scores` is given:

| Key | Formula |
| --- | --- |
| `prompt_share` | `count(domain) / B` |
| `token_share` | `sum(response_mask over that domain's rows) / total_tokens` |
| `prompt_count` | `count(domain)` |
| `mean_response_length` | `domain_tokens / domain_prompts` |
| `reward_abs_mean` | `sum(|r| * mask over domain) / domain_tokens` |

For a 3-D reward the K axis is summed **before** `abs()` (`:133-137`) — so it is
`|sum_K r|`, not `sum_K |r|`. `abs` at all because the OPD reward is signed and a mean near
zero would hide a large magnitude that still drives the gradient (`:135-136`).

`prompt_share` vs `token_share` is the paper's central measurement. The unit test states it
plainly: two prompts, one with 80 response tokens and one with 4 → prompt shares 0.5/0.5,
token shares 80/84 and 4/84 (`test_mt_opd_domain_metrics.py:33-45`).

### 3.4 `compute_domain_loss_weights` — `mt_opd.py:159-335` (M1 + M2)

**Returns `None` when `target_shares is None`** (`:207-208`). That is the default, and it
means the loss is bit-identical to naive routing. Everything below is opt-in.

**M1.** For each domain present in the batch:

```
observed[d] = domain_tokens[d] / total_tokens
raw[d]      = target_shares.get(d, 0.0)              # :225  <-- absent => 0.0
if sum(raw) <= 0:  raw[d] = 1/len(present)           # :226-227
target[d]   = raw[d] / sum(raw)
scale[d]    = target[d] / observed[d]                # :288
```

⚠️ **The docstring at `:188-189` says "Missing domains default to an equal split." The code
gives `0.0`.** Equal split happens only when *every* present domain is missing. Trust the
code.

**M2** (`alpha = normalize_reward_scale`, `True` → 1.0, validated to `[0, 2]` at `:237-238`).
Gradient share is really `token_share × reward_magnitude`, so equalizing tokens alone leaves
the domain with the larger teacher–student gap dominating. Per-domain magnitude `mag[d]`
comes from `reward_scale_stat`:

| `reward_scale_stat` | How `mag` is computed | Notes |
| --- | --- | --- |
| `mean` | `sum(|r| * mask over d) / domain_tokens` (`:262`, `:277`) | length-diluted: per-token `|reward|` correlates with response length at r = −0.87 (math) / −0.80 (code), so `mean` reads math's scale ~3.4× below code/IF (`:250-254`) |
| `q75` / `q90` | `torch.quantile` of the positive masked values (`:287`) | filler-immune, but `torch.quantile` hard-fails above 2²⁴ elements |
| `top10` | mean of the top `numel//10` via `topk` (`:283-284`) | same intent, no size limit |

Then the direction:

```
divide   (default, :316-321):  scale[d] /= mag[d] ** alpha      # skipped when mag <= 1e-8; NO clamp
multiply (:322-325):           fac = (mag[d] / ref[d]) ** alpha
                               fac = clip(fac, 0.05, 20.0)
                               scale[d] *= fac
```

⚠️ **`divide` is the default and the source documents it as measured harmful** (`:290-296`):
over training the easy domain's gap collapses ~35× while math/code collapse ~2×, so
`1/mag**alpha` snowballs onto the *easiest* domain — measured IF loss weight 27 → 91,
gradient token share 57% against a 33% target. `multiply` instead makes the budget follow
the *remaining* gap, so a domain whose gap collapsed loses share automatically. If you turn
M2 on, use `multiply`.

`ref[d]` for `multiply` is either this run's own step-1 magnitude (`reward_scale_anchored`,
so step 1 is exactly M1 and only the drift moves shares) or the running M1-weighted mean
magnitude (`:301-312`).

**Final rescale** (`:332-334`):

```
weights *= total_tokens / sum(weights * per_seq_tokens)
```

so the token-weighted mean of the weights is 1 — total loss magnitude, and therefore the
effective learning rate, is preserved and only the *relative* shares move.

### 3.5 `apply_teacher_conflict_policy` — `mt_opd.py:405-491` (M3)

The MT path evaluates every teacher on every student token and then throws all but the
routed one away. M3 spends that already-paid-for signal. It is the one signal a single
mixed-reward run (MixRL) structurally cannot have: MixRL sees one scalar per domain, never
three opinions on the same token (`:415-418`).

```
top1   = stacked[..., 0]                       # [N, B, T]  the student's top-1 candidate
spread = top1.max(0) - top1.min(0)             # [B, T], nats
thresh = conflict_quantile is None ? conflict_nats
                                   : quantile(spread[mask], conflict_quantile)
agree  = spread <= thresh
```

| `policy` | Behaviour |
| --- | --- |
| `none` (default) | returns the **same** `routed_logprobs` object, `None` weight, empty metrics (`:440-441`) |
| `mask` | token weight = `agree`, then rescaled by `total/kept` so dropping tokens does not shrink the loss (`:476-483`) |
| `consensus` | replaces the target with the across-teacher **mean** where teachers agree, keeps the routed target where they do not (`:485-491`) |

Why a quantile is offered: measured mean spread is **0.131 nats** against a max of **29.2**,
and only **0.70%** of tokens exceed 1.0 nats (`:455-459`, `:521-524`). Conflict is real but
concentrated in a rare tail, so a fixed nats threshold either catches almost nothing or has
to be guessed against a drifting distribution.

### 3.6 `compute_teacher_conflict_metrics` — `mt_opd.py:494-562`

Pure instrumentation, same `spread` definition. Emits `spread_mean`, `contested_frac`,
`spread_max`, plus `{domain}/spread_mean` and `{domain}/contested_frac` when labels are
passed. Empty dict for fewer than two teachers.

---

## 4. `mt_opd.*` config surface

**Every key here is declared in no YAML.** All require Hydra's `+` append form.

| Key | Default | Legal values | Effect | Read at `ray_trainer.py` |
| --- | --- | --- | --- | --- |
| `teacher_domains` | — **required** | `list[str]` | index *i* names teacher *i*: `[0]` = `reward_model`, `[i]` = `mt_reward_model_{i}`. Also the whitelist for target shares | `:662`, raise `:666` |
| `n_additional_teachers` | `0` | int | how many `mt_rm_{i}` groups to spawn; must equal `len(teacher_domains) - 1` | `:663`, used `:1428` |
| `domain_weighting` | `"domain_routing"` | `domain_routing` \| `uniform` | ⚠️ **never validated** — a typo silently selects routing (`mt_opd.py:51`) | `:664` |
| `target_share_domains` + `target_share_values` | `None` | two parallel lists | **the preferred M1 form**; a length mismatch raises | `:675-688` |
| `target_gradient_shares` | `None` | `dict[str, float]` | **M1**. `None` ⇒ loss bit-identical to naive. Ratios need not sum to 1. An unknown domain raises | `:690-702` |
| `normalize_reward_scale` | `False` | `bool` or float ∈ `[0, 2]` | **M2** α. Requires `target_gradient_shares` (raise `:741`). α = 0 ≡ M1 exactly | `:706-717` |
| `reward_scale_stat` | `"mean"` | `mean` \| `q75` \| `q90` \| `top10` | ⚠️ validated only inside the kernel (`mt_opd.py:239`), and only when M1 is on — so a typo can be a permanent silent no-op | `:718` |
| `reward_scale_direction` | `"divide"` | `divide` \| `multiply` | ⚠️ `divide` is the default and is documented as measured harmful (§3.4) | `:719-726` |
| `reward_scale_anchored` | `False` | bool | pins the `multiply` reference at step 1. Requires `multiply` + `mean` | `:727-738` |
| `conflict_policy` | `"none"` | `none` \| `mask` \| `consensus` | **M3** | `:746-753` |
| `conflict_nats` | `1.0` | float | shared threshold for M3 policy and conflict metrics | `:754-756`, `:2177-2183` |
| `conflict_quantile` | `None` | float ∈ (0, 1) | self-calibrating; overrides `conflict_nats`. ⚠️ `0.0` silently becomes `None` (`:758` truthiness) | `:757-765` |

A note on the two M1 forms: the source comment at `:671-673` says a dict-valued Hydra
override for `target_gradient_shares` "does not compose on the executor", which is why the
two-parallel-list form exists. A pure Hydra probe shows the dict literal *does* compose, so
the problem is shell quoting on the way in, not Hydra. Use the list form and avoid the
question.

Related keys, also `+`-only: `+actor_rollout_ref.rollout.log_prob_top_k` (**mandatory**,
see §6), `+actor_rollout_ref.rollout.top_k_strategy`, `+...reward_weight_mode`,
`+...teacher_temperature`, `+...top_p_intersec_p`,
`+mt_reward_model_{i}.{enable,model.path,model.input_tokenizer,model.use_remove_padding,micro_batch_size_per_gpu}`.

---

## 5. Minimal working invocation

The local `scripts/local/mt_opd.sh` emits the following runtime extension keys and
teacher settings. The same form is useful when invoking `main_ppo` directly:

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=token_reward_direct \
  +actor_rollout_ref.rollout.reward_mode=mt_opd \
  +actor_rollout_ref.rollout.log_prob_top_k=256 \
  actor_rollout_ref.model.path=/path/to/mixsft \
  actor_rollout_ref.rollout.name=vllm \
  reward_model.enable=True \
  reward_model.strategy=fsdp \
  reward_model.model.path=/path/to/math-teacher \
  reward_model.model.input_tokenizer=null \
  +mt_opd.teacher_domains=[math,code,if] \
  +mt_opd.n_additional_teachers=2 \
  +mt_reward_model_1.enable=True \
  +mt_reward_model_1.model.path=/path/to/code-teacher \
  +mt_reward_model_1.model.input_tokenizer=null \
  +mt_reward_model_2.enable=True \
  +mt_reward_model_2.model.path=/path/to/if-teacher \
  +mt_reward_model_2.model.input_tokenizer=null \
  +mt_opd.target_share_domains=[math,code,if] \
  +mt_opd.target_share_values=[0.4,0.4,0.2] \
  data.train_files=/path/to/rl_train_with_domain.parquet \
  data.val_files=/path/to/rl_val.parquet \
  custom_reward_function.path=<verl>/verl/utils/reward_score/opd_val_dispatch.py \
  custom_reward_function.name=reward_func
```

Drop the two `target_share_*` lines and you have naive M-OPD — which is exactly what the
shipped launcher produces.

The teacher↔domain binding is **positional**:

```
teacher_domains[0] = "math"  <->  reward_model.model.path
teacher_domains[1] = "code"  <->  mt_reward_model_1.model.path
teacher_domains[2] = "if"    <->  mt_reward_model_2.model.path
```

Only the **count** is checked (`:2121-2126`). Swap two paths and math prompts get distilled
from the code teacher, silently, forever.

---

## 6. Traps, ordered by when they bite you

### 6.1 Before Ray starts

1. **`reward_mode` needs `+`.** `scripts/local/mt_opd.sh:87` and `opd.sh:66` emit it
   without one, while the same script correctly uses `+` for every `mt_opd.*` key. Since
   `reward_mode` is declared nowhere, Hydra refuses:
   `ConfigCompositionException: Could not override 'actor_rollout_ref.rollout.reward_mode'.`
   *Verified by executing a compose-only probe.*

2. **Shipped defaults break the actor and the teachers.**
   `actor_rollout_ref.rollout.tensor_model_parallel_size` defaults to `2`
   (`rollout.yaml:57`) — fails on 1 GPU. `actor.ppo_mini_batch_size` defaults to `256`
   (`actor.yaml:15`) — must be ≤ `data.train_batch_size`. Both
   `ppo_micro_batch_size{,_per_gpu}` are null — `ActorConfig.__post_init__` asserts one is
   set. `reward_model.micro_batch_size_per_gpu` is null with `use_dynamic_bsz=false` →
   `TensorDict.split(None)`.

### 6.2 On the first step

3. **`log_prob_top_k` defaults to 0 on the read path.** No launcher and no YAML sets it;
   `:1968` reads `.get("log_prob_top_k", 0)`. `mt_opd` raises at `:1973-1974`. The `256` in
   `workers/config/rollout.py:154` is the dataclass default and reaches only the vLLM
   engine, not this read. Fix: `+actor_rollout_ref.rollout.log_prob_top_k=256`.

4. **Teacher 0 crashes on `raw_prompt`.** `reward_model.model.input_tokenizer` defaults to
   the student path (`reward_model.yaml:33`), i.e. non-null ⇒
   `_do_switch_chat_template=True` (`fsdp_workers.py:1810-1814`) ⇒ the teacher goes down
   `_switch_chat_template_token_level`, which indexes `non_tensor_batch["raw_prompt"]`
   (`:2513-2519`). That key is absent unless `data.return_raw_chat=True`
   (`legacy_data.yaml:59` defaults False), and `_get_gen_batch`'s keep-set excludes it
   anyway. `mt_opd.sh:107-111` sets `input_tokenizer=null` only for teachers 1..N−1. Fix:
   add `reward_model.model.input_tokenizer=null`.

5. **The `domain` column is mandatory** and no builder under `training/scripts/rl/` emits
   it. `:2110-2114` raises — *after* all N teacher forwards have run, so you pay full
   compute before the error. Build the train parquet with
   `experiments/data/build_mix_rl_raw_union.py`.

6. **`only_stu` is the only safe `top_k_strategy` under MT-OPD.** Routing replaces *only*
   `teacher_on_student_log_probs` (`:2149`); `teacher_top_k_ids` and `overlap_mask` still
   belong to teacher 0. And `top_p_intersec` has no branch in
   `compute_distillation_reward` at all → `UnboundLocalError` at `dp_actor.py:773`, despite
   `workers/config/rollout.py:155` declaring the value.

### 6.3 Silently wrong

7. **Unknown `domain` labels now fail before routing.** The MT-OPD trainer rejects labels
   outside `teacher_domains` before `build_domain_weights`; the standalone helper retains
   its uniform fallback for non-trainer callers.

8. **Teacher↔domain binding is validated at trainer setup.** The trainer requires exactly
   `n_additional_teachers + 1` unique, non-empty domain labels, then preserves positional
   mapping through worker registration and routing.

9. **`conflict_nats` reaches both policy and metric.** The trainer passes
   `self.mt_conflict_nats` at `:2177-2183`, so policy and reported contested fractions use
   the same configured threshold.

10. **M1 + M3-mask changes the loss scale.** Each factor alone has token-weighted mean 1;
    their product at `:2222` is not re-normalized, while both docstrings advertise
    learning-rate preservation.

11. **M3 `mask` does not mask tokens.** The `[B,T]` weight is collapsed to a per-sequence
    mean (`:2218-2221`), so contested tokens inside a kept sequence contribute full
    gradient. The unit tests verify the per-token contract that the integration never
    exercises.

12. **M2 `divide` is the default and is documented as harmful**, with no upper clamp — a
    `mag` of 1e-6 at α=1 multiplies the scale by 1e6.

13. **Anchored M2 can abort mid-run.** The anchor is captured from the first MT batch only
    and filters to the domains present then (`:2189-2201`); a later batch containing a
    domain that was absent raises at `mt_opd.py:245-248`.

14. **`torch.quantile` hard-fails above 2²⁴ elements.** Two reachable sites:
    `mt_opd.py:468` (whenever `conflict_quantile` is set) and `:287` (`q75`/`q90`). At the
    docstrings' own 11570-token math responses, ~1450 math sequences cross it. `top10` uses
    `topk` and is safe.

15. **`actor/pg_clipfrac` is inflated by up to K** in the 3-D branch:
    `masked_mean(values_(B,T,K), mask_(B,T,1))` sums `B·T·K` clipped entries over a `B·T`
    denominator (`core_algos.py:1249`). `pg_clipfrac_lower` is hard-coded `0.0`.

16. **`algorithm.use_kl_in_reward=True` broadcast-fails** against a 3-D
    `token_level_scores` (`:437`), and nothing validates the combination.

17. **`mt_opd/domain/{d}/loss_weight` logs one representative sample** (`dom_w[sel[0]]`) —
    exact under M1/M2, arbitrary once M3-mask varies weights within a domain.

---

## 7. M4 is unreachable

`refresh_opd_advantage` (`mt_opd.py:338-402`) fixes *reward* staleness: the OPD reward is
`-(S_logp − T_on_S) * w(S_logp)`, which depends on the student's own current log-prob, so
after an inner update the stored reward describes a student that no longer exists.
Importance sampling cannot repair this — IS corrects a wrong sampling *distribution*, not a
wrong *reward*. And the fix is free: `T_on_S` comes from a frozen teacher on already-sampled
ids and stays valid forever, while `update_policy` already recomputes `S_logp` on the same
`student_top_k_ids` for the gradient.

It cannot fire. `dp_actor.py:985` and `:1095-1099` both require
`teacher_on_student_log_probs` in the batch, but `ray_trainer.py:3479-3489` pops exactly
that key and `update_actor` is at `:3527` — 38 lines later. So
`mt_opd/m4_advantage_refreshed` (`dp_actor.py:1108`) can never appear in the metrics.
Supporting signs: `refresh_opd_advantage` is absent from `mt_opd.py`'s `__all__`,
`opd_refresh_advantage` exists in `config/actor.py:128` but in no YAML, and the function has
**zero tests**.

To confirm on a live run: check whether `mt_opd/m4_advantage_refreshed` ever appears. It
should not.

---

## 8. What the release actually gives you

No YAML or script in the release sets a single M1/M2/M3/M4 key, nor
`mt_opd.domain_weighting`, `reward_model.reward_manager`, `data.sampler.*`,
`data.domain_weights`, or `custom_reward_function.path=.../opd_val_dispatch.py`. With
`target_gradient_shares=None`, `compute_domain_loss_weights` returns `None` and
`conflict_policy="none"` returns its input unchanged — so **`mt_opd.sh` as shipped is naive
M-OPD, routing only.** The settings behind the paper's headline numbers, and the
`divide`-vs-`multiply` / anchored / `mean`-vs-`q90` / `none`-vs-`mask` choices, are not in
this release. The measured constants quoted throughout the docstrings — response lengths
11570/9032/364, spread mean 0.131 nats with max 29.2 and 0.70% above 1.0, IF weight 27→91,
per-token `|reward|` vs length r = −0.87/−0.80 — come from run logs that are not present
here.

The kernels are complete and unit-tested. The knobs that drive them are yours to set.
