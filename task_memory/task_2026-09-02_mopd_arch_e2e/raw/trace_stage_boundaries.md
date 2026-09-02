# Open-MOPD artifact-level pipeline trace (branch `yc-mopd`, vendored verl `0.7.0.dev`)

## 0. There are exactly three artifact formats in this repo

| Kind | Produced by | Consumed by | Directly loadable? |
|---|---|---|---|
| **HF model dir** (`config.json` + `*.safetensors` + tokenizer) | `verl.model_merger`, `param_merge.py`, HF hub | every trainer's `model.path`, vLLM eval | yes |
| **verl FSDP sharded ckpt** (`model_world_size_{WS}_rank_{R}.pt` + `huggingface/` metadata-only + `fsdp_config.json`) | SFT trainer, RL/OPD/MT-OPD actor | only the *same* verl role for resume | **no — needs `model_merger`** |
| **parquet / jsonl** | data builders, vLLM rollout, RL dumps | trainers, scorers | yes |

Every path is strictly local: `/data/ycfeng/Open-MOPD/training/verl/verl/utils/local_fs.py:13-17` raises on any `"://"`, and upstream `verl/utils/fs.py`+`hdfs_io.py` are **absent** (replaced by `local_fs.py` / `local_io.py`). `trainer.default_hdfs_dir` does not exist in `ppo_trainer.yaml`. **This is an Open-MOPD-specific removal.**

---

## 1. Stage 0 → MixSFT

**Reads.** `/data/ycfeng/Open-MOPD/scripts/local/sft.sh:43-64` → `torchrun -m verl.trainer.sft_trainer`, hydra config `sft_trainer_engine` (`/data/ycfeng/Open-MOPD/training/verl/verl/trainer/sft_trainer.py:378`).

- `data.train_files` = parquet with a **`messages`** column (list of `{role,content}`); optional `tools`, `enable_thinking` (`/data/ycfeng/Open-MOPD/training/verl/verl/utils/dataset/multiturn_sft_dataset.py:64-66,116-128`). Producers: `/data/ycfeng/Open-MOPD/training/scripts/sft/build_weighted_mix.py:15` and `shard_sft_parquet.py:15` both emit `OUTPUT_FIELDS = ["messages","domain","source","dataset","split","difficulty","sample_id"]`.
- `model.path` = base HF dir. `model.tokenizer_path` (`/data/ycfeng/Open-MOPD/training/verl/verl/trainer/config/model/hf_model.yaml:16`) = the **post-training tokenizer package** built by `/data/ycfeng/Open-MOPD/training/scripts/sft/prepare_posttrain_tokenizer.py:170-173` (writes tokenizer files **+ `generation_config.json`**, no weights).
- Load-time contract (**Open-MOPD patch**): `/data/ycfeng/Open-MOPD/training/verl/verl/workers/config/model.py:120-128` prefers `local_tokenizer_path/generation_config.json` over the base model's — this is how the multi-EOS turn-end contract enters training.

**Writes** — `${trainer.default_local_dir}/global_step_{N}/` (`/data/ycfeng/Open-MOPD/training/verl/verl/utils/checkpoint/checkpoint_handler.py:70`):

```
global_step_N/
  model_world_size_{WS}_rank_{R}.pt      # fsdp_checkpoint_manager.py:225,231  (sharded, per rank)
  optim_world_size_{WS}_rank_{R}.pt      #                          :226,236
  extra_state_world_size_{WS}_rank_{R}.pt#                          :227,245  (lr_scheduler + rng)
  fsdp_config.json                       #                          :296-302  {"FSDP_version","world_size"}
  huggingface/                           #                          :258
      config.json                        #                          :280
      generation_config.json             #                          :266-278  (Open-MOPD source hunt)
      tokenizer*.json / chat template     #                          :282
  data_{dp_rank}.pt                      # checkpoint_handler.py:89
latest_checkpointed_iteration.txt        # checkpoint_handler.py:98-102 (parent dir)
```

`huggingface/` holds **metadata only — no weights** (`save_contents` default `["model","optimizer","extra"]`, `/data/ycfeng/Open-MOPD/training/verl/verl/trainer/config/sft_trainer_engine.yaml:44`; weights would need `hf_model`, gated by `should_save_hf_model`, `checkpoint_manager.py:92-98`).

### Blocking defect: you cannot make the SFT stage emit HF weights via `save_contents`

`/data/ycfeng/Open-MOPD/training/verl/verl/workers/engine/fsdp/transformer_impl.py:159-165` passes the checkpoint config under the wrong keyword:

```python
self.checkpoint_manager = FSDPCheckpointManager(
    model=self.module, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
    processing_class=self.model_config.get_processor(),
    checkpoint_contents=self.checkpoint_config,   # <-- swallowed by **kwargs
)
```
`FSDPCheckpointManager.__init__` accepts `checkpoint_config=None, **kwargs` and only rescues the key `"tokenizer"` (`fsdp_checkpoint_manager.py:75-88`), so `checkpoint_config` stays `None` and `BaseCheckpointManager` falls back to the hard-coded defaults (`checkpoint_manager.py:52-57`). The Megatron engine passes it correctly (`/data/ycfeng/Open-MOPD/training/verl/verl/workers/engine/megatron/transformer_impl.py:221`), and so does the PPO actor worker (`/data/ycfeng/Open-MOPD/training/verl/verl/workers/fsdp_workers.py:892`) — **only the FSDP SFT engine path is broken.** Consequence: `checkpoint.save_contents=[...,hf_model]` is silently ignored → merging is mandatory after MixSFT. Workaround that exists in-tree: the legacy entry point `verl.trainer.fsdp_sft_trainer` honors `trainer.checkpoint.save_contents` (`/data/ycfeng/Open-MOPD/training/verl/verl/trainer/fsdp_sft_trainer.py:579-600`); note it writes `data.pt`, not `data_{dp_rank}.pt` (`:559`). This looks like an upstream verl kwarg-rename regression, not an MOPD change (no MOPD marker on those lines).

---

## 2. MixSFT → domain RL: the mandatory merge step

**Can stage 2 read stage 1's output directly? No.** The RL actor calls `AutoModelForCausalLM.from_pretrained(local_path)` (`/data/ycfeng/Open-MOPD/training/verl/verl/workers/fsdp_workers.py:404-409`, `local_path = copy_to_local(self.config.model.path)` at `:818`). A `global_step_N/` dir has no weights file HF can read.

**The one merge helper:** `/data/ycfeng/Open-MOPD/training/scripts/sft/merge_model.sh` (name says `sft/`, it is backend-generic).

```
bash training/scripts/sft/merge_model.sh <CKPT_DIR> [OUT_DIR]
  :34  requires <CKPT_DIR>/fsdp_config.json
  :38  default OUT = dirname(CKPT)/merged_hf/basename(CKPT)
  :46-50  python -m verl.model_merger merge --backend fsdp --local_dir CKPT --target_dir OUT
  :51-53  --trust-remote-code when TRUST_REMOTE_CODE=1 (default 1)
  :58  asserts OUT/config.json exists
```

Merger internals — the input contract is *two* sibling artifacts:
- `--local_dir/fsdp_config.json` → `world_size` (`/data/ycfeng/Open-MOPD/training/verl/verl/model_merger/fsdp_model_merger.py:75-87`); `FSDP_version` is written but never read.
- `--local_dir/huggingface/` → hard-coded as `hf_model_config_path` (`/data/ycfeng/Open-MOPD/training/verl/verl/model_merger/base_model_merger.py:142`), used for `AutoConfig` (`:194`), `GenerationConfig` (`:234`), `hf_processor`/`hf_tokenizer` (`:317-318`).
- Shards loaded as `model_world_size_{WS}_rank_{R}.pt` for `R in range(total_shards)` (`fsdp_model_merger.py:149`), DTensor-aware, **cast to bf16 unconditionally** (`:169,181`). `mesh_dim_names` must be `("fsdp",)` or `("ddp","fsdp")`; FSDP+TP raises `NotImplementedError` (`:120,200`).

**Output** = plain HF dir: `config.json`, `generation_config.json`, `model.safetensors` (or `model-0000X-of-0000N.safetensors` + `model.safetensors.index.json` for a 3B bf16 model), tokenizer files (`base_model_merger.py:313,321,324`).

Alternative documented in-tree: point `--target_dir` at `<ckpt>/huggingface` to promote the metadata dir into a full HF model in place (`/data/ycfeng/Open-MOPD/training/verl/recipe/open_math_reasoning/README.md:47`). It works because the config is read in `__init__` before any write.

`python -m verl.model_merger test --backend fsdp --local_dir ... --test_hf_dir ...` validates key/shape/dtype/values against a reference (`fsdp_model_merger.py:229-260`).

---

## 3. Domain RL x3 (math / code / if)

**Reads** (`/data/ycfeng/Open-MOPD/scripts/local/rl.sh:44-66`, `-m verl.trainer.main_ppo`, config `ppo_trainer`, `/data/ycfeng/Open-MOPD/training/verl/verl/trainer/main_ppo.py:36`):
- `actor_rollout_ref.model.path` = merged MixSFT **HF dir**.
- `data.train_files` / `data.val_files` = RL parquet: `data_source`, `prompt` (list of `{role,content}`), `ability`, `reward_model{style,ground_truth}`, `extra_info{index}` — exact arrow schema at `/data/ycfeng/Open-MOPD/training/scripts/rl/build_dapo_math_rl_dataset.py:62-76`. Siblings: `build_deepcoder_rl_dataset.py`, `build_nemotron_if_rl_dataset.py`, `build_livecodebench_rl_val_dataset.py`.
- `reward_model.enable=False`; rule-based scoring via `custom_reward_function.path/name` if given (`rl.sh:68-73`).
- Whole-row passthrough: `RLHFDataset.__getitem__` returns the full parquet row (`/data/ycfeng/Open-MOPD/training/verl/verl/utils/dataset/rl_dataset.py:338,497`) and `collate_fn` splits tensors vs `non_tensor_batch` (`:83-111`). **Any extra parquet column becomes a `non_tensor_batch` key** — that is the mechanism behind `domain` in stage 4.

**Writes** (`/data/ycfeng/Open-MOPD/training/verl/verl/trainer/ppo/ray_trainer.py:1512-1558`) — note the extra `actor/` level:

```
${trainer.default_local_dir}/
  global_step_N/
    actor/                      # :1521  <- the merge unit
      model_world_size_*.pt, optim_*.pt, extra_state_*.pt
      fsdp_config.json
      huggingface/              # config + generation_config + tokenizer (weights only if hf_model in save_contents)
    data.pt                     # :1548  (single file, not per-dp-rank)
  latest_checkpointed_iteration.txt   # :1554-1558
```

`actor_rollout_ref.actor.checkpoint.save_contents` defaults to `['model','optimizer','extra']` (`/data/ycfeng/Open-MOPD/training/verl/verl/trainer/config/actor/actor.yaml:111`) and **is honored here** (`fsdp_workers.py:892`), so for RL/OPD you *can* append `hf_model` and skip the merger — rank-0 then writes real weights into `global_step_N/actor/huggingface/` (`fsdp_checkpoint_manager.py:308-356`).

Optional side artifacts: `trainer.rollout_dump.{enable,dir,every_n_steps,max_steps}` (`ppo_trainer.yaml:241-252`) → `<step>.gen_{round:03d}.jsonl`; `trainer.validation_data_dir` (`:178`) → `<step>.jsonl`; writer at `ray_trainer.py:888-900`. These JSONL rows are the input to `/data/ycfeng/Open-MOPD/experiments/data/rft_from_rollouts.py`.

**Boundary answer:** OPD cannot read `global_step_N/actor` directly. Merge with `--local_dir <...>/global_step_N/actor` (passing `global_step_N` fails `merge_model.sh:34`, since `fsdp_config.json` lives one level deeper).

---

## 4. OPD (single teacher) and MT-OPD (three teachers)

Both are `verl.trainer.main_ppo` with `algorithm.adv_estimator=token_reward_direct`.

**Reads — every model input is an HF dir:**

| Hydra key | Artifact | Loader |
|---|---|---|
| `actor_rollout_ref.model.path` | merged MixSFT HF dir (student) | `fsdp_workers.py:404` |
| `reward_model.model.path` | teacher **0** HF dir | `fsdp_workers.py:1808,1837` |
| `+mt_reward_model_{i}.model.path`, i=1..N-1 | teachers 1..N-1 HF dirs | merged over `config.reward_model` at `ray_trainer.py:1436`, missing key → hard error `:1431-1435` |
| `+mt_opd.teacher_domains`, `+mt_opd.n_additional_teachers` | domain names, teacher count | `ray_trainer.py:661-666` |
| `actor_rollout_ref.rollout.reward_mode` | `opd_kl` / `delta_opd` / `exopd` / `mt_opd` | `ray_trainer.py:88` |

`RewardModelWorker._build_model` uses **`AutoModelForCausalLM`** (`fsdp_workers.py:1837`) even though its own docstring still claims `AutoModelForTokenClassification` (`:1737`) — an Open-MOPD patch with a stale docstring; teachers are ordinary causal LMs, not scalar RMs.

**Data contract added by MT-OPD:** `non_tensor_batch['domain']` is required; absent → hard error at `ray_trainer.py:2109-2113`, and its cardinality must equal `len(mt_opd.teacher_domains)` (`:2121-2126`). None of `training/scripts/rl/build_*` emits a `domain` column; the two builders that do are `/data/ycfeng/Open-MOPD/experiments/data/build_mix_rl_raw_union.py:57-66` (top-level `pa.field("domain", pa.string())` **and** `extra_info.domain`) and `build_mix_rl_union.py:104-108`. **So the MT-OPD train parquet must come from an `experiments/data/build_mix_rl_*_union.py` run, not from the per-domain RL builders.**

Also required: `actor_rollout_ref.rollout.log_prob_top_k > 0` for `delta_opd`/`mt_opd` (`ray_trainer.py:1973-1974`; default is 256 at `/data/ycfeng/Open-MOPD/training/verl/verl/workers/config/rollout.py:154`).

**Writes:** identical layout to stage 3 (`global_step_N/actor/...` + `data.pt` + tracker) — same code path.

### Two launcher-level issues at this boundary

1. **`reward_mode` is likely rejected by Hydra as written.** The key exists in **no** composed yaml (`grep -c reward /data/ycfeng/Open-MOPD/training/verl/verl/trainer/config/rollout/rollout.yaml` → 0 matches over 326 lines) and is **not** a `RolloutConfig` field (only a comment mentions it, `workers/config/rollout.py:159`). It is read then *deleted* under `open_dict` precisely so the strict dataclass conversion survives (`ray_trainer.py:86-92`). Yet `/data/ycfeng/Open-MOPD/scripts/local/opd.sh:66` and `/data/ycfeng/Open-MOPD/scripts/local/mt_opd.sh:87` pass it **without** a leading `+`, while the same scripts correctly use `+mt_opd.*` / `+mt_reward_model_N.*` (`mt_opd.sh:92-93,107-112`). Under Hydra's default struct mode that asymmetric form should fail with "Could not override … use +…". **Unverified by execution** — `hydra` is not installed in this environment, so I could not run a composition test. Reproducers should try `+actor_rollout_ref.rollout.reward_mode=...` if the launcher errors out.
2. **Teacher 0 re-templates, teachers 1..N-1 do not.** `reward_model.model.input_tokenizer` defaults to `${actor_rollout_ref.model.path}` (non-null, `/data/ycfeng/Open-MOPD/training/verl/verl/trainer/config/reward_model/reward_model.yaml:33`), which sets `_do_switch_chat_template=True` (`fsdp_workers.py:1810-1813`) and routes teacher 0 through `_switch_chat_template_token_level` (`:2691`). `mt_opd.sh:109` sets `input_tokenizer=null` **only** for the additional teachers; `opd.sh` never sets it at all. For same-family teachers the intended value is `null`. I report the wiring divergence, not a measured numeric effect.

---

## 5. Eval

**Reads** (`/data/ycfeng/Open-MOPD/scripts/local/eval.sh:45-60` → `/data/ycfeng/Open-MOPD/evals/rollout_engine/vllm_rollout.py`):
- `--model` = **merged HF dir**. `GenerationConfig.from_pretrained(model_path)` supplies `stop_token_ids` when `--stop-token-ids` is omitted (`vllm_rollout.py:109-119`) — this closes the chain: post-train tokenizer `generation_config.json` → ckpt `huggingface/generation_config.json` (`fsdp_checkpoint_manager.py:266-278`) → merged HF `generation_config.json` (`base_model_merger.py:234`) → eval stop ids.
- `--input` = one or more eval parquets needing a `prompt` column (chat list, JSON string, or raw text; `:275, 152-171`) and, for per-dataset output naming, a `dataset` column (`:306-320`).

**Writes** `<output-dir>/{dataset}_rollouts_base{B}[_offset{O}]_rank{R}.parquet` (`:304,318,322`) = every input column plus `completion_index`, `completion`, `completion_tokens`, `finish_reason`, `stop_reason`, `generated_special_token_counts`, and the echoed sampling params.

**Scoring — two entry points over the same parquets:**
- `/data/ycfeng/Open-MOPD/evals/verifier/score.py` — `--rollout <parquet...> [--output scores.json]`, dispatches on the `dataset` column (`:27-31,76-97`), writes `{"results":[…],"macro_official","macro_live"}` (`:190-199`). Denominators come from `--data-dir/<dataset>.parquet` (`common/utils.py:69-73`), default `evals/data/parquet` (`common/utils.py:15`).
- `/data/ycfeng/Open-MOPD/evals/score_rollouts.py` — `--rollout-dir --dataset --out`, adds a per-row `score` column and **overwrites the rollout parquets in place** (`:224-227`), plus a summary JSON with truncation rate (`:229-236`).
- Extra row columns the scorers need: `answer` (AIME, `score_functions/math/aime.py:26-27`), `metadata` with LCB testcases (`score_rollouts.py:97-106`), `metadata.instruction_id_list`/`kwargs` or the flat equivalents (IF, `score_functions/instruction_following/common.py:123-129`), `request_id` (denominator count).

**Gaps in the released eval path** (documented in `/data/ycfeng/Open-MOPD/evals/README.md`, absent on disk — verified by `ls`): `evals/rollout_engine/build_data.py` (README:32), `evals/rollout_engine/scripts/run_all.sh` (README:142), `evals/misc/fetch_livecodebench_assets` (README:76), `evals/verifier/third_party/repos.lock.json` + `patches/` (README:84-85), and the whole `evals/data/` tree (parquet inputs + `benchmark_assets/livecodebench/`). `/data/ycfeng/Open-MOPD/evals/aggregate_lcb_scorecard.py:11-31` is hard-coded to internal run names (`20260624_{model}_lcb_sharded`, labels like `"0623d S40 (rcorr2.0)"`) and is not a general tool. **Net: eval parquets are the reproducer's own responsibility; only `vllm_rollout.py → score.py / score_rollouts.py` is runnable as shipped.**

---

## 6. Direct answer: can stage N+1 consume stage N's output?

| Boundary | Direct? | Required action |
|---|---|---|
| MixSFT → domain RL | **No** | `merge_model.sh <ckpt>/global_step_N` (or `verl.model_merger merge --backend fsdp`). `save_contents=[...,hf_model]` does **not** work here (`transformer_impl.py:164` bug) |
| domain RL → OPD/MT-OPD (as teacher) | **No** | merge `<ckpt>/global_step_N/**actor**`; or pre-empt it with `actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model']` → weights land in `global_step_N/actor/huggingface/` |
| MixSFT → OPD/MT-OPD (as student) | **No** | same merged MixSFT HF dir as stage 3 uses |
| OPD/MT-OPD → eval | **No** | merge `global_step_N/actor` → HF dir for vLLM |
| any stage → *resume itself* | **Yes** | `trainer.resume_mode=auto` reads `latest_checkpointed_iteration.txt` from `default_local_dir` (`checkpoint_manager.py:167-197`; `ray_trainer.py:1560-1610`). Default is `auto` (`ppo_trainer.yaml:200`) — **reusing one output dir across stages silently resumes instead of starting fresh** |
| merged HF dirs → parameter-merge baselines | **Yes** | `/data/ycfeng/Open-MOPD/experiments/backend/param_merge.py:39-60` reads `model.safetensors.index.json` (or a bare `model.safetensors`) |
| Megatron backend (present, unused by the launchers) | **No** | `dist_ckpt/` + `huggingface/` + `transformer_config.json` (`megatron_utils.py:513-527`; `megatron_checkpoint_manager.py:365-481`); merge with `--backend megatron`, `local_dir/dist_ckpt` read at `megatron_model_merger.py:494` |

## 7. Attribution and open items

**Open-MOPD-specific at these boundaries:** `local_fs.py`/`local_io.py` replacing `fs.py`/`hdfs_io.py`; the tokenizer-package generation-config preference (`workers/config/model.py:120-128` and `fsdp_checkpoint_manager.py:266-277`); `RewardModelWorker` loading `AutoModelForCausalLM` with a `dtype` knob (`fsdp_workers.py:1830-1841`); `reward_mode` pop-and-route (`ray_trainer.py:86-92`); `mt_reward_model_{i}` worker registration (`ray_trainer.py:1426-1440,1492-1494`); `mt_opd.*` validation block (`ray_trainer.py:660-767`); `verl/workers/actor/mt_opd.py`; `trainer.rollout_dump`; `_ensure_validation_data_source` (`ray_trainer.py:95-123`); `training/scripts/sft/prepare_posttrain_tokenizer.py`; `experiments/data/build_mix_rl_*_union.py`; the whole `evals/` tree.

**Upstream verl, untouched at these boundaries:** `verl/model_merger/*` (all four files read as pure upstream), `checkpoint_manager.py`, `checkpoint_handler.py`, the FSDP checkpoint file naming, and the `checkpoint_contents=` kwarg bug.

**Explicitly unverified / unclear:**
1. Whether Hydra actually rejects `actor_rollout_ref.rollout.reward_mode=` without `+` — no `hydra` in this env; the config-schema evidence is definitive, the runtime behavior is inferred.
2. `fsdp_config.json`'s `FSDP_version` field is written but never consumed by any merger — dead metadata.
3. `evals/verifier/scripts/manage_third_party.py` exists but its target `evals/verifier/third_party/repos.lock.json` does not, so I could not determine what commits it would pin.
4. No script in the repo chains the stages; the four launchers are independent and each requires the operator to run `merge_model.sh` between them. The only place the merge is documented is `merge_model.sh`'s own usage text — it is **not** mentioned in `/data/ycfeng/Open-MOPD/README.md` (whose Quick Start says `--model /path/to/mixsft --teacher /path/to/math-teacher`, i.e. already-merged dirs) nor in `/data/ycfeng/Open-MOPD/scripts/local/README.md`. That omission is the single biggest practical trap for a reproducer.