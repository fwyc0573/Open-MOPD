# Open-MOPD Architecture — Reading Guide

## Modification History

| Date       | Summary of Changes                                                        |
| ---------- | ------------------------------------------------------------------------- |
| 2026-09-02 | First version. Built from a 12-subsystem deep read, adversarially verified, then re-vetted claim-by-claim (43/48 confirmed; corrections folded in). |

**Scope of this document.** Where the code lives, who calls whom, and what crosses each
boundary. The MT-OPD algorithm itself is in
[`02_mt_opd_algorithm.md`](02_mt_opd_algorithm.md). Running it is in
[`03_reproduce_e2e.md`](03_reproduce_e2e.md).

**Confidence.** Every claim below carries a `file:line`. Line numbers were checked
against source; residual drift is ≤2 lines. Claims that could not be settled by reading
were settled by executing a probe, and are marked **[probed]**.

---

## 0. Orientation in one paragraph

Open-MOPD trains **one** student to absorb **three** independently RL-trained domain
teachers (math / code / instruction-following). At every training step the student
samples responses; then *every* teacher scores the student's own top-K candidate tokens;
then each prompt's `domain` label selects which teacher's scores actually count. The
selected teacher's per-token log-probs become a **dense `[B, T, K]` reward** rather than a
scalar, and that reward flows straight through as the advantage. On top of that routing,
Open-MOPD adds a per-sequence weight that forces a *requested* per-domain share of the
optimization budget. The reason that weight is needed at all is the paper's diagnosis:
under `loss_agg_mode=token-mean` a domain's gradient share follows its **token** share,
not its prompt share — and measured response lengths were math 11570 / code 9032 / IF 364,
so IF held 20% of the prompts but 0.9% of the tokens
(`training/verl/verl/workers/actor/mt_opd.py` module docstring, and
`training/verl/tests/test_mt_opd_domain_metrics.py:1-12`).

The novelty is concentrated in exactly two places:

| Where | What |
| --- | --- |
| `training/verl/verl/workers/actor/mt_opd.py` (562 lines, 7 pure functions) | the mechanisms, as dependency-free math on tensors |
| `training/verl/verl/trainer/ppo/ray_trainer.py:2100-2230` | the only production caller — routing, overwrite, weighting, metrics |

Everything else is vendored verl `0.7.0.dev` (`training/verl/verl/version/version`) plus
patches.

---

## 1. Layer map

| Layer | Directory | Entry symbol | Reads config? |
| --- | --- | --- | --- |
| **Launchers** — dry-run-by-default shell | `scripts/local/` (`common.sh`, `sft.sh`, `rl.sh`, `opd.sh`, `mt_opd.sh`, `eval.sh`) + root `eval.sh` shim | `local_run_command` (`common.sh:226`) | no — it *builds* the override list |
| **Trainer driver** — single-controller Ray | `training/verl/verl/trainer/` | `RayPPOTrainer.fit` (`ppo/ray_trainer.py:1691`) | yes, Hydra |
| **Worker layer** — FSDP + vLLM as Ray actors | `training/verl/verl/workers/` | `ActorRolloutRefWorker` (`fsdp_workers.py:161`), `RewardModelWorker` (`fsdp_workers.py:1735`) | dataclass-converted |
| **Algorithm kernels** — pure tensor math | `workers/actor/mt_opd.py`, `trainer/ppo/core_algos.py` | `compute_domain_loss_weights`, `compute_token_reward_direct_advantage` | no |
| **Data + reward** | `training/scripts/{sft,rl}/`, `experiments/data/`, `verl/utils/dataset/`, `verl/utils/reward_score/`, `verl/workers/reward_manager/` | `default_compute_score`, `NaiveRewardManager` | partially |
| **Evals** — standalone, no Hydra | `evals/` (`rollout_engine/` + `verifier/`), `experiments/backend/`, `experiments/analysis/` | `evals.rollout_engine.vllm_rollout`, `evals.verifier.score` | argparse only |

The layers are genuinely separable: the kernels import nothing from verl, and the evals
import nothing from the trainer except three code-grading helpers
(`evals/score_rollouts.py:96-101`).

---

## 2. The four-stage recipe, and what it actually is in code

```
  MixSFT                    3 x domain RL                 MT-OPD                    eval
  ------                    -------------                 ------                    ----
  torchrun -m               python -m                     python -m                 python -m
  verl.trainer.sft_trainer  verl.trainer.main_ppo          verl.trainer.main_ppo     evals.rollout_engine
                            adv_estimator=grpo             adv_estimator=            .vllm_rollout
                                                            token_reward_direct        |
                                                           reward_mode=mt_opd          v
                                                                                     python -m
                                                                                     evals.score_rollouts
       |                          |                             |
       v                          v                             v
  FSDP shards               FSDP shards                   FSDP shards
       |                          |                             |
       +-- merge_model.sh --------+------ merge_model.sh -------+---> loadable HF dir
```

**There is one entrypoint for three different jobs.** `verl.trainer.main_ppo` is RL, OPD,
and MT-OPD; the difference is two config keys. That is the single most important structural
fact about this codebase:

| Job | `algorithm.adv_estimator` | `actor_rollout_ref.rollout.reward_mode` | `reward_model.enable` |
| --- | --- | --- | --- |
| domain RL (GRPO) | `grpo` | — | `False` |
| single-teacher OPD | `token_reward_direct` | `opd_kl` | `True` |
| delta-OPD | `token_reward_direct` | `delta_opd` | `True` |
| **MT-OPD** | `token_reward_direct` | `mt_opd` | `True` |

`reward_mode` is read at `ray_trainer.py:88` and then **deleted** from the config
(`:86-92`, under `open_dict`). The delete is load-bearing: `RolloutConfig`
(`verl/workers/config/rollout.py`) has no such field, so the dataclass conversion in the
worker would reject it. **[probed]** — the key exists in no YAML under
`verl/trainer/config/`, which is why it must be appended with Hydra's `+` form; see §6.

---

## 3. Module catalog

Legend: **★** Open-MOPD addition · **◐** upstream file with Open-MOPD patches · ○ upstream

### 3.1 Algorithm kernels

**★ `training/verl/verl/workers/actor/mt_opd.py`** — no verl imports, no state, no
`DataProto`. Pure functions on tensors, which is why the whole test suite for it runs on
CPU in seconds.

| Symbol | Line | Returns | One-line contract |
| --- | --- | --- | --- |
| `build_domain_weights` | 24 | `[B,N]` f32 | domain label → one-hot teacher row. **An unknown label does not raise** — it gets uniform `1/N` (`:57-61`) |
| `select_routed_teacher_logprobs` | 65 | `[B,T,K]` | stacks N teacher tensors to `[N,B,T,K]`, contracts with `[N,B,1,1]`. Returns a NEW tensor |
| `compute_domain_share_metrics` | 90 | `dict[str,float]` | the diagnosis: `prompt_share` vs `token_share` per domain |
| `compute_domain_loss_weights` | 159 | `[B]` f32 **or `None`** | M1+M2. `None` when `target_shares is None` (`:207-208`) ⇒ loss bit-identical to naive |
| `refresh_opd_advantage` | 338 | `[B,T,K]` | M4. **Unreachable from `fit()`** — see §7 |
| `apply_teacher_conflict_policy` | 405 | `(target, weight\|None, metrics)` | M3. `policy="none"` returns the *same object* (`:440-441`) |
| `compute_teacher_conflict_metrics` | 494 | `dict` | free instrumentation; empty dict for <2 teachers (`:525-526`) |

Two documentation defects inside this file, both worth knowing before you trust a
docstring here:

- `:188-189` says "Missing domains default to an equal split." The code assigns `0.0`
  (`:225-227`). Equal split happens only when *every* present domain is missing from
  `target_shares` (`:328-334`).
- `:290-296` records `reward_scale_direction="divide"` as **measured harmful** (IF loss
  weight 27→91, gradient share 57% against a 33% target) — and `divide` is nonetheless
  the default (`:166`). Only `multiply` clamps, to `[0.05, 20]` (`:316-325`).

**◐ `training/verl/verl/trainer/ppo/core_algos.py`**

- `@register_adv_est("token_reward_direct")` at `:876`. Body (`:895-902`, inside
  `torch.no_grad()`): `advantages = token_level_rewards * response_mask`, with
  `response_mask.unsqueeze(-1)` applied only inside `if token_level_rewards.dim() == 3:`
  (`:897-898`) — i.e. only for the dense top-K case. `returns = advantages.clone()`.
- `token_reward_direct_plus_grpo` at `:905` returns a **3-tuple** (its type annotation
  still says 2), adding a GRPO outcome term scattered onto the sampled top-K slot
  (`_scatter_outcome_adv_onto_sampled_topk`, `:970`). The caller tolerates it via a
  `len(res) == 3` check.
- The 3-D policy-loss branch is `:1214-1251`; it sums over the K axis at `:1240`. Note
  `pg_clipfrac_lower` is hard-coded `0.0` (`:1251`), and there is an unguarded per-micro-batch
  `print` at `:1215`.
- `token_grpo` (`:352`) is **registered but unreachable**: its signature has no `**kwargs`
  while the caller always forwards `true_reward_score` (`ray_trainer.py:548-561`,
  `:2486-2487`) ⇒ `TypeError`. No config in the repo selects it.

### 3.2 Trainer driver

**◐ `training/verl/verl/trainer/ppo/ray_trainer.py`** — the file to read if you read only
one. Landmarks:

| Lines | What |
| --- | --- |
| `86-92` | `_pop_direct_opd_rollout_options` — reads `rollout.reward_mode`, then deletes it |
| `95-131` | `_ensure_validation_data_source` — backfills `data_source` from `dataset` for validation rows |
| `657-658` | raises when an OPD mode is set without `reward_model.enable=True` |
| `660-765` | the mt_opd config parse — 13 `OmegaConf.select` reads, 10 `ValueError` sites |
| `766-777` | the non-mt_opd else branch — sets 11 of 12 `self.mt_*` siblings, **omits `mt_reward_scale_stat`** (latent `AttributeError`, harmless only because its sole read at `:2208` is inside the mt_opd branch) |
| `1145-1165` | `_get_gen_batch` — the keep-set at `:1149-1151` includes `"domain"`, so domain stays on `batch` and is **absent from `gen_batch`** in sync mode |
| `1372-1510` | `init_workers` — `mt_rm_{i}` registration `:1427-1440`, `self.mt_rm_wgs` `:1492-1496` |
| `1512-1558` | `_save_checkpoint` — the `global_step_N/actor/` layout |
| `1691` | `fit()` |
| `2100-2230` | **the MT-OPD reward block** — see `02_mt_opd_algorithm.md` §2 |
| `3478-3489` | pops `teacher_on_student_log_probs` + 6 siblings |
| `3527` | `update_actor` — 38 lines *after* the pop, which is what makes M4 unreachable |

**★ `trainer/ppo/kl_controller.py`** (14 lines) — `update_kl_loss_coef_from_reward`, a
multiplicative sign-of-reward controller that mutates `actor_config.kl_loss_coef` in place
(`ray_trainer.py:3505-3520`) and ships it in `meta_info["actor_kl_loss_coef"]`. Its default
tracked key is `"delta_opd/weighted_reward_mean"`, so it is inert under `mt_opd`. Also
note `actor.yaml` sets `kl_loss_coef: 2.5` equal to `adaptive_kl_loss_max_coef: 2.5`
(`:85`, `:92`) — the controller starts pinned at its own ceiling. Both `use_kl_loss` and
`adaptive_kl_loss_coef` default `false`, so this only bites when you turn both on.

**★ `trainer/ppo/validation_sampling.py`** — `build_validation_repeat_plan`, for the
heterogeneous val suite (AIME mean@64 / LCB mean@10 / IF mean@1).

### 3.3 Worker layer

**◐ `training/verl/verl/workers/fsdp_workers.py`** — the key patch is that
`RewardModelWorker` (`:1735`) is repurposed from a **scalar reward model** into a **teacher
language model**:

- `_build_model` (`:1801-1885`) loads `AutoModelForCausalLM`. The class docstring still
  says `AutoModelForTokenClassification` — stale, ignore it.
- `_compute_teacher_top_k_log_probs` (`:1911-2015`) chunks at 1024 rows and returns teacher
  log-probs **gathered at the student's token ids**, plus overlap masks. The `torch.gather`
  path covers `{only_stu, union, union-intersection, top_p_intersec}`; `intersection`
  instead does a masked max over the teacher's own top-K (`:1997-2009`), giving `-inf`
  where a student token falls outside it.
- `compute_rm_score` (`:2664-2871`) returns **no `rm_scores` key when `top_k > 0`**
  (`:2825-2828`). The reward math is deliberately deferred to the actor worker, because
  only the actor holds the student's log-probs.
- `compute_distillation_reward` is registered here (`:1091-1093`) and **only** here.

**◐ `workers/actor/dp_actor.py`**

| Lines | What |
| --- | --- |
| `241-545` | `_forward_micro_batch` — top-K gather, Ulysses re-alignment |
| `601-774` | `compute_distillation_reward` — the OPD reward kernel. `only_stu` branch (`:710-714`): `rm_scores = -(student_logp − teacher_on_student) * softmax_K(weights)` |
| `773` | where `top_k_strategy=top_p_intersec` throws `UnboundLocalError` — no branch handles it, despite `workers/config/rollout.py:155` declaring the value |
| `777`, `797` | `compute_delta_opd_reward`, `compute_exopd_reward` |
| `839-858` | `_optimizer_step` — **skips `optimizer.step()` on a non-finite grad norm** |
| `944-1225` | `update_policy` — `select_keys` `:975-976`, M4 gate `:1095-1099`, **M1 application `:1128-1132`** |

`:1128-1132` is where the whole domain-weighting mechanism lands: the `[B]` weight is
multiplied into **`advantages`**, not into `response_mask`, with a rank-generic
`view(-1, *([1]*(dim-1)))`. Consequence: `ppo_kl` and `pg_clipfrac` stay unweighted.

**Megatron cannot run any OPD mode.** `compute_distillation_reward` is absent from
`workers/megatron_workers.py`, whose `compute_rm_score` (`:1268-1275`) just forwards
`self.rm.compute_reward`. Under `mt_opd` it dies at `ray_trainer.py:2106` with a `KeyError`
on `teacher_on_student_log_probs`, before the routing code is even reached.

### 3.4 Data + reward

| Module | Mark | Role |
| --- | --- | --- |
| `experiments/data/build_mix_rl_raw_union.py` | ★ | the MT-OPD union builder — 6-column RL parquet with `domain`, plus a 13-field unified `extra_info` so three domains share one Arrow type (`:40-66`) |
| `experiments/data/build_mix_rl_union.py` | ★ | restores a legacy 24K selection by greedy subsequence join. The selection template it needs is produced by nothing in-repo ⇒ **unrunnable as released** |
| `training/scripts/rl/build_*.py` | ★ | per-domain 5-column RL parquets |
| `training/scripts/sft/build_weighted_mix.py` | ★ | MixSFT builder; released mix 459646/459646/229823 (2:2:1). **Also emits `domain`** |
| `training/scripts/sft/prepare_posttrain_tokenizer.py` | ★ | grafts a post-training chat template onto an unchanged base vocab |
| `utils/dataset/rl_dataset.py` | ◐ | `__getitem__` (`:338`) returns the whole row; `collate_fn` (`def` at `:83`, body `:96-111`) funnels every non-tensor column into `non_tensor_batch`. **This is how an arbitrary `domain` column reaches the trainer** |
| `utils/dataset/domain_weighted_sampler.py` | ★ | hardcodes `DOMAIN_ORDER = ("math","code","if")` (`:15`); raises without a `domain` column (`:57-59`); **zero production callers** — `ray_trainer.py:1797` imports only `weighted_quotas` from it |
| `utils/reward_score/opd_val_dispatch.py` | ★ | the validation reward router. Two predicates + a fallback: `_is_if` → `instruction_following`, `_is_code` → `rllm_code_reward`, everything else (incl. math) → `ttrl_math`. Collapses every scorer's output to `{score, acc}` so mixed-domain validation keeps a uniform per-row schema |
| `workers/reward_manager/naive.py` | ◐ | passes `rm_scores` through verbatim as the training reward; files the rule score under `reward_extra_info["true_reward_score"]` (`:280`). Note `reward_extra_info` is **rebound** at `:279`, discarding the per-row rule metrics |
| `trainer/ppo/code_reward_pool.py` | ★ | the shared subprocess/Ray testcase pool — the real code-reward engine |

Note on `domain`: several builders emit it — `build_mix_rl_raw_union.py:64`,
`build_mix_rl_union.py:108`, `training/scripts/sft/build_weighted_mix.py:15`, and others.
The RL-only per-domain builders under `training/scripts/rl/` do **not**, which is the part
that matters: a parquet built by those alone will fail MT-OPD at `ray_trainer.py:2110-2114`.

### 3.5 Evals

| File | Lines | Role |
| --- | --- | --- |
| `evals/rollout_engine/vllm_rollout.py` | 415 | multi-process data-parallel offline generation, one `CUDA_VISIBLE_DEVICES` slice per rank (`:335-340`) |
| `evals/verifier/score.py` | 208 | exact-set dataset registry (`:27-31`); reports `official` + `live` rates + macros |
| `evals/score_rollouts.py` | 244 | **rewrites the rollout parquets in place** with a `score` column (`:218`, `:223-226`); substring dataset matching. Its per-row code graders swallow every exception to `0.0` (`:106-107`, `:140-141`) — a missing LiveCodeBench checkout stamps `score=0.0` everywhere and exits 0 |
| `experiments/backend/param_merge.py` | 344 | average / task-arithmetic merge baseline |
| `experiments/analysis/capability_subspace.py` | 405 | SVD subspace + DACSP gating. **Zero callers outside its own test**, whose own suite documents a negative result (`test_capability_subspace.py:454-465`) |

---

## 4. Data contracts

### 4.1 SFT parquet — column is `messages`, not `question`/`answer`

This trips everyone. `scripts/local/sft.sh` launches `-m verl.trainer.sft_trainer`, and
that module's Hydra entry is `@hydra.main(config_name="sft_trainer_engine")`
(`verl/trainer/sft_trainer.py:378`). So `verl/trainer/config/sft_trainer.yaml`, with its
`prompt_key: question` / `response_key: answer`, belongs to a **different** trainer
(`fsdp_sft_trainer.py`) and is not in play. **[probed]** — passing `data.prompt_key=question`
to `sft_trainer_engine` raises `ConfigCompositionException: Could not override
'data.prompt_key'`.

`sft_trainer.py:393` always builds `MultiTurnSFTDataset` unless `data.custom_cls.path` is
set. That class reads its column name from `config["multiturn"]["messages_key"]`
(`utils/dataset/multiturn_sft_dataset.py:62-66`), while `sft_trainer_engine.yaml` declares
`messages_key` **flat** under `data:` (`:24-27`) — a block the nested lookup never sees.
Net effect: `messages_key`, `tools_key`, and `enable_thinking_key` are all dead config, and
the live column name is the hard default `"messages"`.

Released schema (`build_weighted_mix.py:135-145`), 7 flat columns, zstd, written as
`train-{00000..}.parquet` + `summary.json`:

```
messages    list<struct<role: string, content: string>>
domain      string
source      string
dataset     string
split       string
difficulty  string
sample_id   string
```

### 4.2 RL prompt parquet — 5 columns, plus `domain` for MT-OPD

```
data_source   string
prompt        list<struct<role: string, content: string>>
ability       string
reward_model  struct<style: string, ground_truth: string>
extra_info    struct<...>          # shape differs per domain
domain        string               # MT-OPD only, and MANDATORY there
```

Per-domain `extra_info` shapes differ (math `{index}`; code `{split, index, source_config}`;
IF 7 fields including `instruction_id_list`, `instruction_kwargs_json`, `family`). The
MT-OPD union normalizes them to one 13-field struct
(`build_mix_rl_raw_union.py:40-66`): `split, index, source_config, sample_id, raw_prompt,
instruction_id_list, instruction_kwargs_json, dataset, agent_name, family, domain,
source_index, mix_index`. `index` is rewritten to `f"{domain}-{mix_index}"`; `source_index`
is `str()`-coerced so math UUIDs and code ints share a type.

`domain` values must be a subset of `mt_opd.teacher_domains` — **and a mismatch does not
raise.** An unlisted label silently gets the average of all teachers, and zero gradient if
M1 is on. See `02_mt_opd_algorithm.md` §5.

### 4.3 Rollout output parquet (eval)

Path: `<output-dir>/{dataset}_rollouts_base{B}[_offset{O}]_rank{R}.parquet`, with a
fallback of `{input_stem}_rollouts_*` when no `dataset` column exists.

Columns = every input column, plus stamped `temperature`, `top_p`, `top_k`, `max_tokens`
(and `stop_token_ids` when supplied), plus per-completion `completion_index`, `completion`,
`completion_tokens`, `finish_reason`, `stop_reason`, `generated_special_token_counts`. One
row per (input row × completion). A row whose `result.outputs` was empty is written without
the last three keys (`vllm_rollout.py:279-285`); `pd.DataFrame.from_records` back-fills them
as NaN from sibling rows, so true raggedness only shows up when *every* row was empty.

### 4.4 Score output JSON — two scorers, two shapes

`evals/verifier/score.py`:

```json
{"results": [{"dataset": ..., "metric": ..., "scored_rows": ..., "correct": ...,
              "official_total": ..., "official": ..., "live": ..., "official_extra": ...,
              "trunc_rows": ..., "repeat_rows": ..., "trunc_repeat_rows": ...,
              "trunc_rate": ..., "repeat_rate": ..., "trunc_repeat_rate": ...}],
 "macro_official": 0.0, "macro_live": 0.0}
```

`official` = correct rows ÷ the complete dataset parquet size. `live` = correct rows ÷ rows
actually present in this run. An **unscored** dataset yields a ragged row: `correct`,
`official`, `live` all null, a `note`, and no `metric`, no `official_extra`, and none of the
six trunc/repeat keys.

`evals/score_rollouts.py` writes a different, flat shape:
`{dataset, rows, metric, correct, scored_rows, total, pct[, truncated, truncation_rate, max_tokens]}`
— and **nothing at all** for an unrecognized dataset name.

### 4.5 Checkpoint layout — and why stage N+1 cannot read stage N

```
SFT   <default_local_dir>/global_step_N/
        model|optim|extra_state_world_size_{WS}_rank_{R}.pt     <- FSDP shards
        fsdp_config.json          {"FSDP_version", "world_size"}  (only world_size is read)
        huggingface/              config.json + tokenizer + generation_config.json
                                  METADATA ONLY — no weights
        data_{dp_rank}.pt
      <default_local_dir>/latest_checkpointed_iteration.txt

PPO   identical, but one level deeper under global_step_N/actor/,
      and a single data.pt instead of per-dp-rank files
```

Evidence: `utils/checkpoint/fsdp_checkpoint_manager.py:225-227, :258, :296`;
`checkpoint_handler.py:70, :86-89, :98`; `ray_trainer.py:1515-1521, :1548`.

Three artifact kinds exist: loadable HF dirs, verl FSDP shards (**not** loadable), and
parquet/jsonl. Everything is local-only — `utils/local_fs.py:13-17` raises on any `"://"`.

| Boundary | Direct? | Conversion needed |
| --- | --- | --- |
| MixSFT → domain RL / OPD student | **No** | `bash training/scripts/sft/merge_model.sh <ckpt>/global_step_N [out]` |
| domain RL → OPD teacher | **No** | same, but on `global_step_N/actor` (one level deeper; passing `global_step_N` fails the `fsdp_config.json` check at `merge_model.sh:34`) |
| OPD/MT-OPD → eval | **No** | merge first, then `--model <merged dir>` |
| any stage → resume itself | **Yes** | `trainer.resume_mode=auto` (the default) reads `latest_checkpointed_iteration.txt`. ⚠️ **Reusing one output dir across stages silently resumes instead of starting fresh** |
| merged HF → `param_merge.py` | Yes | reads `model.safetensors.index.json` or a bare `model.safetensors` |
| RL rollout JSONL → RFT SFT parquet | Yes | `cat step*.jsonl \| python -m experiments.data.rft_from_rollouts extract` (stdin only) |

**The shortcut that does not work.** `checkpoint.save_contents=[...,'hf_model']` would let
a stage write HF weights directly and skip merging. It works for the PPO actor
(`fsdp_workers.py:892`) and for Megatron, but **not for the SFT stage**:
`workers/engine/fsdp/transformer_impl.py:164` passes the config as `checkpoint_contents=`,
which `FSDPCheckpointManager.__init__` swallows into `**kwargs`, leaving
`checkpoint_config=None` and the hard-coded three defaults in force
(`fsdp_checkpoint_manager.py:75-96`, `:308`). A grep for `checkpoint_contents=` finds
exactly this one site. So merging after MixSFT is mandatory, and no README mentions it.

---

## 5. Reading order

Fifteen files, in the order that makes each one make sense given the previous.

| # | File / range | Why here |
| --- | --- | --- |
| 1 | `scripts/local/mt_opd.sh` | the exact override set — and what is *not* configured |
| 2 | `verl/workers/actor/mt_opd.py` (all 562 lines) | the whole novelty, dependency-free; the docstrings carry the measured numbers |
| 3 | `verl/trainer/ppo/ray_trainer.py:86-92, 657-777` | how `reward_mode` and the 13 `mt_opd.*` keys are parsed and validated |
| 4 | `ray_trainer.py:2100-2230` | the MT-OPD reward block — every kernel call in production order |
| 5 | `ray_trainer.py:1145-1165` | why `domain` survives rollout (the keep-set) |
| 6 | `verl/workers/fsdp_workers.py:1911-2015, 2664-2871` | teacher scoring; why there is no `rm_scores` when `top_k>0` |
| 7 | `verl/workers/actor/dp_actor.py:601-774` | the OPD reward kernel that MT-OPD reuses unchanged |
| 8 | `dp_actor.py:944-1225` | `select_keys`, the 3-D forward, the M4 gate, the M1 application, `_optimizer_step` |
| 9 | `verl/trainer/ppo/core_algos.py:876-1020, 1214-1251` | `token_reward_direct` and the 3-D policy loss |
| 10 | `ray_trainer.py:1372-1510` | worker-group fan-out; `mt_rm_{i}` registration and colocation |
| 11 | `experiments/data/build_mix_rl_raw_union.py:33-115` | the on-disk `domain` contract |
| 12 | `training/verl/tests/test_mt_opd_gradient_share.py` + `test_mt_opd_m2_m3.py` | the executable spec for M1/M2/M3, including the advantage-scaling ≡ mask-scaling proof at `:132-155` |
| 13 | `training/verl/tests/trainer/ppo/test_mt_opd_batch_routing.py` | the domain-plumbing spec; `:58` asserts `"domain" not in gen_batch.non_tensor_batch` |
| 14 | `verl/workers/reward_manager/naive.py:119-295` | how `rm_scores` becomes the advantage, and where the rule score goes |
| 15 | `evals/verifier/score.py:59-121` + `evals/score_rollouts.py:168-241` | the two competing scorers and where they disagree |

---

## 6. Config surface

`ppo_trainer.yaml` composes from group defaults: `actor@actor_rollout_ref.actor: dp_actor`,
`data@data: legacy_data`, `ref@actor_rollout_ref.ref: dp_ref`,
`rollout@actor_rollout_ref.rollout: rollout`, `model@actor_rollout_ref.model: hf_model`,
`critic@critic: dp_critic`, `reward_model@reward_model: dp_reward_model`.

Some keys the code reads are declared in **no** YAML and are therefore reachable only via
Hydra's `+` append form. Getting this wrong is a composition error, not a silent default:

| Key | Declared? | Form required |
| --- | --- | --- |
| `actor_rollout_ref.rollout.reward_mode` | **no** | `+actor_rollout_ref.rollout.reward_mode=mt_opd` |
| `actor_rollout_ref.rollout.log_prob_top_k` | **no** (the `256` in `workers/config/rollout.py:154` is the dataclass, a different object) | `+actor_rollout_ref.rollout.log_prob_top_k=256` |
| every `mt_opd.*` key | **no** | `+mt_opd.teacher_domains=[math,code,if]` |
| every `mt_reward_model_{i}.*` key | **no** | `+mt_reward_model_1.model.path=...` |
| `teacher_ref_reward_model.*`, `...top_k_strategy`, `...reward_weight_mode`, `...teacher_temperature`, `...top_p_intersec_p`, `...exopd_lambda`, `actor_rollout_ref.actor.opd_refresh_advantage`, `data.gen_batch_size`, `data.domain_weights.*`, `algorithm.filter_groups.domain_weights` | **no** | `+` |

The full `mt_opd.*` table (defaults, legal values, where each is read) is in
[`02_mt_opd_algorithm.md`](02_mt_opd_algorithm.md) §4.

Declared keys whose defaults will break a small run:

| Key | Default | Why it breaks |
| --- | --- | --- |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `2` (`rollout.yaml:57`) | fails on 1 GPU |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `256` (`actor.yaml:15`) | must be ≤ `data.train_batch_size` |
| `actor_rollout_ref.actor.ppo_micro_batch_size{,_per_gpu}` | both `null` | `ActorConfig.__post_init__` asserts one is set |
| `reward_model.micro_batch_size_per_gpu` | `null` with `use_dynamic_bsz=false` | `TensorDict.split(None)` |
| `reward_model.model.input_tokenizer` | `${actor_rollout_ref.model.path}`, i.e. **non-null** (`reward_model.yaml:33`) | non-null ⇒ `_do_switch_chat_template=True` ⇒ indexes `non_tensor_batch["raw_prompt"]`, absent unless `data.return_raw_chat=True` |
| `actor_rollout_ref.actor.loss_agg_mode` | `token-mean` | this is M1's entire premise — under token-mean, gradient share == token share |
| `trainer.save_freq` / `test_freq` | `-1` | setting them to `0` is a `ZeroDivisionError` mid-loop (`sft_trainer.py:323-324` takes the modulo before the `>0` guard) |

---

## 7. Gaps in the release

Read this section before you tune anything.

### 7.1 The two launchers that cannot run as released **[probed]**

`scripts/local/opd.sh:66` and `scripts/local/mt_opd.sh:87` emit
`actor_rollout_ref.rollout.reward_mode=<mode>` **without** a leading `+`, while the same
`mt_opd.sh` correctly uses `+` for every `mt_opd.*` key. Since `reward_mode` is declared
nowhere, Hydra refuses:

```
ConfigCompositionException: Could not override 'actor_rollout_ref.rollout.reward_mode'.
To append to your config use +actor_rollout_ref.rollout.reward_mode=mt_opd
```

This fires at composition time — before Ray starts, before any GPU is touched. Add the `+`.

Second, no launcher and no YAML sets `log_prob_top_k`, and `ray_trainer.py:1968` reads it
with a default of `0`, so `mt_opd` and `delta_opd` raise at `:1973-1974`. Add
`+actor_rollout_ref.rollout.log_prob_top_k=<K>`.

Third, `mt_opd.sh` sets `input_tokenizer=null` and `param_offload=True` only for teachers
`1..N-1` (`:107-111`); **teacher 0** inherits the non-null default and hits the
`raw_prompt` path. Add `reward_model.model.input_tokenizer=null`.

### 7.2 The shipped `mt_opd.sh` is naive M-OPD, not Open-MOPD

No YAML or script in the release sets a single M1/M2/M3/M4 key, nor
`mt_opd.domain_weighting`, `reward_model.reward_manager`, `data.sampler.*`,
`data.domain_weights`, or `custom_reward_function.path=.../opd_val_dispatch.py`. With
`target_gradient_shares=None`, `compute_domain_loss_weights` returns `None`
(`mt_opd.py:207-208`) and the loss is bit-identical to routing-only distillation. The
settings behind the paper's headline numbers are not in this release, and neither are the
run logs the docstrings quote (response lengths 11570/9032/364; spread mean 0.131 nats;
IF weight 27→91).

### 7.3 Half-wired

1. **M4 (`opd_refresh_advantage`) is unreachable from `fit()`.** `dp_actor.py:985` and
   `:1095-1099` both require `teacher_on_student_log_probs` in the batch, but
   `ray_trainer.py:3479-3489` pops exactly that key and `update_actor` is at `:3527`. So
   `mt_opd/m4_advantage_refreshed` can never be emitted. The function also has zero tests.
2. `self.mt_reward_scale_stat` missing from the non-mt_opd else branch (`:766-777`).
3. Megatron cannot run any OPD mode (§3.3).
4. `mt_teacher_{i}_on_student_log_probs` is neither popped nor in `select_keys` — dropped
   by `data.select`, but transported across the driver→worker dispatch at
   `(N-1) × [B,T,K]` per step.
5. `token_grpo` registered and unreachable (§3.1).
6. `DomainWeightedSampler` has zero production callers.
7. `capability_subspace.py` has zero callers; `orthonormal_basis` (`:140`) uses unpivoted
   `torch.linalg.qr`, so its docstring's "rank-revealing" claim does not hold.
8. `algorithm.mask_truncated_samples` / `overlong_filter` / `norm_adv_by_std_in_grpo` are
   inert for the OPD estimators (forwarded only inside the hard-coded GRPO branch,
   `ray_trainer.py:533`). `clip_advantages` clamps `advantages` but not `returns`.
9. `NaiveRewardManager.compute_true_reward` (`naive.py:83`) is a constructor-only flag with
   no Hydra key and no caller, so the `:126-131` fast path is unreachable: the expensive
   rule pass always runs, and its metrics are then discarded by the rebind at `:279`.
10. `actor_rollout_ref.rollout.kl_estimator` is fully dead — read at `ray_trainer.py:1970`,
    assigned to an unused local at `dp_actor.py:614`, and undeclared.

### 7.4 Documented but absent

All verified missing, and absent from `git ls-files`:

- `evals/rollout_engine/build_data.py` (`evals/README.md:31-34`) — **there is no eval-input
  builder anywhere in the repo**; the eval parquet schema is only inferable from its readers.
- `evals/misc/fetch_livecodebench_assets` (README:76)
- `evals/rollout_engine/scripts/run_all.sh` (README:140-143)
- `evals/verifier/third_party/{repos.lock.json, patches/, repos/}` (README:92-96) — so all
  four `manage_third_party` commands raise `FileNotFoundError`, even `--help`
  (`manage_third_party.py:103` reads the lock file before `parse_args()`)
- the whole `evals/data/` tree
- the 24K selection template `build_mix_rl_union.py` restores from

Doc drift: `experiments/backend/README.md` documents `--model`/`--output`, which do not
exist (real names `--input NAME=PATH` / `--output-dir`) and omits the two required
`--mode`/`--reference`; `--base` *is* real (`param_merge.py:296`).
`evals/README.md:116-118` claims `max_num_seqs=2048` / `max_num_batched_tokens=8192`; the
code defaults are 1024 / None. `evals/verifier/README.md:13` claims `score.py` is the only
scoring entrypoint; `evals/score_rollouts.py` ships.
`training/install_requirements.sh` installs neither vllm nor flash-attn (both extras-only),
while all three PPO launchers hard-code `rollout.name=vllm`. torch does arrive, but only
transitively via `accelerate`/`peft`/`torchdata`/`tensordict` (`setup.py:26-45`) — so its
version and CUDA build are uncontrolled.
