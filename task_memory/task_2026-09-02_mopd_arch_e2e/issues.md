# Issues, Blockers, and Open Questions

## Modification History

| Date       | Summary of Changes                                        |
| ---------- | --------------------------------------------------------- |
| 2026-09-02 | Initial: release defects found, environment issues, open items |
| 2026-09-03 | Closed r5 runtime questions; recorded non-core test failures and final evidence |

---

## Part A — Defects found in the Open-MOPD release

These are findings about the repository, not about our test harness. Severity is "will a
reproducer hit this": **BLOCKING** = the documented command cannot run; **SILENT** = it runs
and gives you something other than what you asked for.

### A1 · BLOCKING · `opd.sh` and `mt_opd.sh` are rejected by Hydra

`scripts/local/opd.sh:66` and `scripts/local/mt_opd.sh:87` emit
`actor_rollout_ref.rollout.reward_mode=<mode>` with no leading `+`. That key is declared in
no YAML under `training/verl/verl/trainer/config/` and is absent from the `RolloutConfig`
dataclass (`training/verl/verl/workers/config/rollout.py`), so Hydra refuses to override it.

**Root cause.** The key is intentionally undeclared: `ray_trainer.py:86-92` reads it and
then `del`s it under `open_dict`, precisely so the worker-side dataclass conversion does not
see an unknown field. An undeclared key can only be *appended*, with `+`. The same
`mt_opd.sh` gets this right for all 13 `mt_opd.*` keys and wrong for this one.

**Evidence (executed twice, independently).**

```
ConfigCompositionException: Could not override 'actor_rollout_ref.rollout.reward_mode'.
To append to your config use +actor_rollout_ref.rollout.reward_mode=mt_opd
```

**Fix.** One character in each of two files:
`+actor_rollout_ref.rollout.reward_mode=${REWARD_MODE}`.

**Status.** Reported here; the launcher files themselves are **not modified** — that is
outside the scope the user set for this task. Our stage scripts reproduce the failure and
then use the corrected direct invocation.

### A2 · BLOCKING · `log_prob_top_k` is never set, and MT-OPD requires it

`ray_trainer.py:1968` reads `self.config.actor_rollout_ref.rollout.get("log_prob_top_k", 0)`.
No launcher and no YAML sets it, so the value is 0 and `:1973-1974` raises
`reward_mode=mt_opd requires actor_rollout_ref.rollout.log_prob_top_k > 0`.

The `256` at `workers/config/rollout.py:154` is a decoy: that is the dataclass default, a
different object, and it reaches only the vLLM engine — never this read.

**Fix.** `+actor_rollout_ref.rollout.log_prob_top_k=256` (or whatever K).

### A3 · BLOCKING · teacher 0 is configured differently from teachers 1..N−1

`mt_opd.sh:107-111` sets `model.input_tokenizer=null`, `model.use_remove_padding=True`, and
`model.fsdp_config.param_offload=True` for each `+mt_reward_model_{i}`. Teacher 0 —
`reward_model.model.*` — gets none of that, so it inherits
`input_tokenizer: ${actor_rollout_ref.model.path}` from `reward_model.yaml:33`, which is
non-null. Non-null sets `_do_switch_chat_template=True` (`fsdp_workers.py:1810-1814`), which
routes the teacher through code that indexes `non_tensor_batch["raw_prompt"]` (`:2513-2519`).
That key is absent unless `data.return_raw_chat=True` (`legacy_data.yaml:59` defaults False),
and `_get_gen_batch`'s keep-set excludes it anyway.

**Fix.** `reward_model.model.input_tokenizer=null`.

### A4 · BLOCKING · MixSFT → any next stage needs an undocumented merge

Every verl checkpoint under `global_step_N/` is FSDP shards plus a `huggingface/` subdir
holding config and tokenizer but **no weights**. `training/scripts/sft/merge_model.sh` is the
conversion, and it is referenced by neither `README.md` nor `scripts/local/README.md`. Its
own `--help` is the only documentation.

The shortcut that would avoid it — `checkpoint.save_contents=[...,'hf_model']` — works for
the PPO actor and for Megatron but **not for the SFT stage**:
`workers/engine/fsdp/transformer_impl.py:164` passes the config as `checkpoint_contents=`,
which `FSDPCheckpointManager.__init__` swallows into `**kwargs`, leaving
`checkpoint_config=None` and the hard-coded defaults in force. A repo-wide grep for
`checkpoint_contents=` finds exactly this one site.

### A5 · BLOCKING · shipped defaults break a single-GPU or small-batch run

| Key | Default | Failure |
| --- | --- | --- |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `2` (`rollout.yaml:57`) | fails on 1 GPU |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `256` (`actor.yaml:15`) | raises when > `data.train_batch_size` |
| `actor_rollout_ref.actor.ppo_micro_batch_size{,_per_gpu}` | both `null` | `ActorConfig.__post_init__` asserts |
| `reward_model.micro_batch_size_per_gpu` | `null` (with `use_dynamic_bsz=false`) | `TensorDict.split(None)` |
| `trainer.save_freq` / `test_freq` | `-1` | setting **`0`** is a `ZeroDivisionError` mid-loop (`sft_trainer.py:323-324` takes the modulo before the `>0` guard) |

### A6 · BLOCKING · `install_requirements.sh` installs no rollout backend

All three PPO launchers hard-code `rollout.name=vllm`, and the installer never requests
`vllm` or `flash-attn` (both extras-only in `setup.py:50,52`). torch does arrive, but only
transitively via `accelerate`/`peft`/`torchdata`/`tensordict` (`setup.py:26-45`), so its
version and CUDA build are uncontrolled. `numpy<2.0.0` is pinned, which downgrades a
numpy-2 host.

### A7 · SILENT · the `domain` column is mandatory and no RL builder emits it

`ray_trainer.py:2110-2114` raises when `non_tensor_batch["domain"]` is absent — but only
**after** every teacher's forward pass has already run, so you pay full compute first. No
script under `training/scripts/rl/` emits the column; use
`experiments/data/build_mix_rl_raw_union.py`.

Worse, a domain label that is *present but unknown* does **not** raise: it gets a uniform
`1/N` teacher row (`mt_opd.py:57-61`), i.e. distillation from the *average* of all teachers,
and under M1 it gets target share `0.0` (`:225`) — **zero gradient**, with the other domains
silently scaled up. The only trace is an extra `mt_opd/domain/<label>/*` metric series.

### A8 · SILENT · teacher↔domain binding is positional and unverified

Only the teacher **count** is checked (`ray_trainer.py:2121-2126`). Swap two `--teacher`
paths and math prompts are distilled from the code teacher, silently, for the whole run.

### A9 · SILENT · `mt_opd.conflict_nats` does not reach the metric

`ray_trainer.py:2177-2183` calls `compute_teacher_conflict_metrics` with only
`teacher_logprobs`, `response_mask`, `domains`, so the metric threshold stays at the kernel
default of 1.0 (`mt_opd.py:498`). Tune `conflict_nats` and the *policy* threshold moves while
`mt_opd/conflict/contested_frac` does not.

### A10 · SILENT · M2's default direction is the one the source calls harmful

`reward_scale_direction` defaults to `"divide"` (`mt_opd.py:166`), and `:290-296` records
`divide` as measured harmful: the easy domain's teacher gap collapses ~35× while math/code
collapse ~2×, so `1/mag**alpha` snowballs onto the easiest domain — measured IF loss weight
27 → 91, gradient token share 57% against a 33% target. `divide` also has no upper clamp,
while `multiply` clamps to `[0.05, 20]` (`:324`).

### A11 · SILENT · M3 `mask` does not mask tokens, and breaks the LR-preservation claim

The kernel returns a `[B,T]` token weight, but `ray_trainer.py:2218-2221` collapses it to a
per-sequence mean before folding it into `domain_loss_weight`. So contested tokens inside a
kept sequence contribute full gradient. The unit tests verify the per-token contract that
the integration never exercises. Separately, each of `dom_w` and the M3 weight has
token-weighted mean 1 on its own, but their product at `:2222` is not re-normalized — while
both docstrings advertise learning-rate preservation.

### A12 · SILENT · docstring contradicts code on missing target shares

`mt_opd.py:188-189`: "Missing domains default to an equal split." The code assigns `0.0`
(`:225-227`); equal split happens only when *every* present domain is missing.

### A13 · Dead or unreachable code

| What | Why unreachable | Evidence |
| --- | --- | --- |
| **M4** `refresh_opd_advantage` | `dp_actor.py:985`/`:1095-1099` need `teacher_on_student_log_probs`; `ray_trainer.py:3479-3489` pops it and `update_actor` is at `:3527` | zero tests, absent from `__all__`, `opd_refresh_advantage` in no YAML |
| `token_grpo` adv estimator | signature has no `**kwargs`, caller always forwards `true_reward_score` | `core_algos.py:352-360` vs `ray_trainer.py:548-561` |
| `top_k_strategy=top_p_intersec` | no branch in `compute_distillation_reward` → `UnboundLocalError` at `dp_actor.py:773`, despite being a declared legal value | `workers/config/rollout.py:155` |
| `DomainWeightedSampler` | zero production callers; `ray_trainer.py:1797` imports only `weighted_quotas` | `data.sampler.class_path` null in both declarations |
| all OPD modes under Megatron | `compute_distillation_reward` absent from `megatron_workers.py`; dies at `ray_trainer.py:2106` with `KeyError` | `megatron_workers.py:1268-1275` |
| `self.mt_reward_scale_stat` in the non-mt_opd branch | set for 11 of 12 siblings; latent `AttributeError` | `ray_trainer.py:766-777` vs `:718` |
| `rollout.kl_estimator` | read, assigned to an unused local, undeclared | `ray_trainer.py:1970`, `dp_actor.py:614` |
| `capability_subspace.py` | zero callers outside its own test, whose suite documents a negative result | `test_capability_subspace.py:454-465` |

### A14 · Documented but absent from the release

`evals/rollout_engine/build_data.py`, `evals/misc/`, `evals/rollout_engine/scripts/run_all.sh`,
`evals/verifier/third_party/`, `evals/data/` — all referenced by `evals/README.md`, none
present. Consequence: **there is no eval-input parquet builder anywhere in the repo**, so the
eval input schema is inferable only from its readers. `manage_third_party.py` raises
`FileNotFoundError` even for `--help` (`:103` reads the lock file before `parse_args()`).

Also unrunnable: `experiments/data/build_mix_rl_union.py`, whose 24K selection template is
produced by nothing in-repo and is not among the released HF artifacts.

Doc drift: `experiments/backend/README.md` documents `--model`/`--output`, which do not
exist, and omits the required `--mode`/`--reference` (`--base` *is* real,
`param_merge.py:296`). `evals/README.md:116-118` claims `max_num_seqs=2048` /
`max_num_batched_tokens=8192`; the code defaults are 1024 / None.
`evals/verifier/README.md:13` claims `score.py` is the only scoring entrypoint;
`evals/score_rollouts.py` ships and **rewrites your rollout parquets in place**.

### A15 · SILENT · reusing one output directory across stages resumes instead of restarting

`trainer.resume_mode` defaults to `auto`, which reads `latest_checkpointed_iteration.txt`. If
two stages share a `default_local_dir`, the second silently resumes the first.

---

## Part B — Environment issues hit in this session

### B1 · `/data` is Ceph RBD and parallel agent fan-out starves the installer

`df -hT` → `/data` on `/dev/rbd1` (xfs), 88% used, 127G free. While two verification
workflows were running (29 + 10 agents) the host showed `%Cpu(s): 90.3 wa` and load average
**235**, and `uv`'s wheel-unpack rate fell to **32 MB/min** — a ~4 GB remainder would have
taken over two hours.

Contributing cause of my own making: polling with `du -sb` on a ~300k-file cache tree, and
`ls -la` on `/data/ycfeng/tmp` (12 000+ entries).

**Resolution:** poll the log file with `grep` only; do not walk the cache tree; let the
fan-out drain before starting large installs. Rate recovered afterwards.

### B2 · `conda env list` and NFS-cache probes time out

`conda env list` and two `find`/`ls` probes into
`/data/ycfeng/hf_home_cudagraph_padding_20260727/hub/...` each exceeded 120 s. That is why
the dummy model family is **fully synthesized** rather than reusing the cached Qwen3
tokenizer — a decision the user confirmed.

### B3 · pip works without a proxy

The internal mirror `http://mirrors.i.basemind.com/pypi/simple/` resolves and serves
directly; `eval $(curl -s http://deploy.i.shaipower.com/httpproxy)` was not needed.

---

## Part C — Open questions

| # | Question | How to settle |
| --- | --- | --- |
| C1 | Does the corrected MT-OPD invocation actually complete a step on a tiny dummy model, and do all three domains appear in `mt_opd/domain/*`? | pending — GPU stage `60_mt_opd.sh` |
| C2 | Does A3 (teacher-0 `raw_prompt` KeyError) actually fire, and does `input_tokenizer=null` remove it? | run `60b` once with and once without the override |
| C3 | Does A13's M4 claim hold at runtime — is `mt_opd/m4_advantage_refreshed` truly never emitted? | grep the `60b` log for that key |
| C4 | Does the vLLM engine accept a 568-vocab / 128-hidden Qwen3? | pending — first `40_rl.sh` run |
| C5 | Does `merge_model.sh` produce a dir that vLLM can load, for a model this small? | pending — `35_merge.sh` then `70_eval.sh --model <merged>` |
| C6 | Is A4's `hf_model` defect observable? | after `30_sft.sh`, `ls global_step_*/huggingface/` and confirm no `*.safetensors` |

## Part D — BLOCKER: no GPU is reachable, and the CPU master is I/O-starved

Raised 2026-09-02 23:35. Work stopped here per the Blocking Issues protocol.

### D1 · All three GPU quota groups are unavailable

| Group | Tag | Result | Evidence |
| --- | --- | --- | --- |
| `codesign` | `h800` | **queued indefinitely** | `Waiting in queue "codesign-default". Insufficient GPU quota. Request: 1 GPU (H800). Queue remaining: H800=0.` — repeated 12× over 12 min |
| `infer_af_test` | `h200` | **submission refused** | `quota-manager-webserver ... denied the request: group[infer_af_test] 已被锁定,禁止提交新任务,请联系管理员` |
| `b300_train_infra` | `b300` | **cross-zone, unusable here** | `workspace is in zone shai-cn-shanghai-sj but target group b300_train_infra is in zone shai-cn-qingyang-cm: cross-zone launch without --image is not allowed` |

Note that `rlaunch --predict-only` reported 9 candidate H800 nodes with 2–6 free GPUs each.
Predict-only checks **node capacity, not quota**, so it is not a gate for this failure — a
useful thing to know for next time.

The B300 route is doubly blocked: cross-zone requires `--image`, which means the workspace
NFS is not auto-mounted, which means the persistent venv at `/data/ycfeng/envs/openmopd-py312`
would not exist on the worker. Using it would require rebuilding the whole environment
inside a container image.

**State right now.** RJob `ws-56153d316be61e0f-jlaunch-nhb5b` is left **queued** on
`codesign`, wrapped in `timeout 10800` (3 h). It will start on its own the moment H800 quota
frees, and it runs the complete chain unattended via
`e2e/run_on_worker.sh`. Nothing needs to be re-launched if quota returns within that window.

### D2 · The CPU master cannot finish even the unit tests

An unrelated 25-file pytest run belonging to another session is running on this same host
(`.../stepfun-performance-optimization/Frontier`, PID 2139279). With both active the host
shows `%Cpu(s): 78–95 wa`, and my own pytest (PID 2116413) has sat in uninterruptible disk
sleep (`D`) for **31 minutes** without finishing collection — it has emitted exactly one
line of output.

Root cause is the same as B1: `/data` is a single Ceph RBD device at 88% capacity, shared by
every process on the host, and a cold `import transformers` alone measured **464 s** on it.

### D3 · Impact on the deliverables

| Requirement | Status |
| --- | --- |
| R1 architecture docs | **complete and verified** — unaffected by the blocker |
| R2.2 dummy models, no weight download | **complete** — 4 × 0.441 M-param Qwen3 + synthesized tokenizer, built and on disk |
| R2 config-level E2E | **complete** — 11/11 Hydra compose cases as expected, in the real venv |
| R2 unit tests | blocked by D2 |
| R2 GPU chain (SFT → merge → RL → OPD → MT-OPD → eval) | blocked by D1 |
| R2.3/R2.4 reproduction guide | scripts are written and syntax-clean; the observed-values section needs D1 resolved |

### D4 · Options

1. **Wait for H800 quota.** Zero cost, no new decisions. The queued job self-starts and
   completes the whole chain. Unknown wait.
2. **Ask the administrator to unlock `infer_af_test`** (the H200 group). The workload needs
   1 GPU for a few minutes; any modern GPU suffices.
3. **Run the training stages on CPU with `rollout.name=hf`.** `verl/utils/device.py:39-51`
   falls back to `"cpu"` when CUDA is absent, and `fsdp_workers.py:506,627` branch on
   `rollout.name == "hf"`, a pure-transformers rollout path. This would exercise the driver
   loop, MT-OPD teacher routing, the per-domain metrics, and the actor update — everything
   except the vLLM rollout, the multi-teacher GPU memory layout, and weight sync. It needs
   the host to be less I/O-contended than it is right now, and it is lower fidelity than
   what was agreed.
4. **Build the environment into a Docker image and use B300 cross-zone.** Highest cost;
   only worth it if both H800 and H200 stay unavailable for a long time.


| # | Question | How to settle |
| --- | --- | --- |
| C1 | Does the corrected MT-OPD invocation actually complete a step on a tiny dummy model, and do all three domains appear in `mt_opd/domain/*`? | **closed** — r5 naive and M1 each completed 2 steps and emitted all three domains |
| C2 | Does A3 (teacher-0 `raw_prompt` KeyError) actually fire, and does `input_tokenizer=null` remove it? | **closed for this path** — corrected teacher runs completed without the KeyError; the shipped mismatch remains documented as A3 |
| C3 | Does A13's M4 claim hold at runtime — is `mt_opd/m4_advantage_refreshed` truly never emitted? | **closed** — no M4 metric appears in corrected MT-OPD logs |
| C4 | Does the vLLM engine accept a 568-vocab / 128-hidden Qwen3? | **closed** — RL and eval vLLM engines initialized and generated successfully |
| C5 | Does `merge_model.sh` produce a dir that vLLM can load, for a model this small? | **closed** — merge wrote safetensors/config/tokenizer and eval loaded the local model |
| C6 | Is A4's `hf_model` defect observable? | **closed** — SFT checkpoint `huggingface/` has metadata/tokenizer only; merge supplies `model.safetensors` |

## Part D — Final verification notes

The final r5 worker's optional repo suite remains non-zero (`111 passed, 8 failed`).
Seven failures are GRM fixture/environment failures and one is a capability-subspace
numerical tolerance mismatch. They are not MOPD core blockers, and no unrelated source
files were changed to hide them. `dummy_math` also has no shipped scorer, so verifier
output is intentionally `unknown_dataset` with `scored_rows=0`.
