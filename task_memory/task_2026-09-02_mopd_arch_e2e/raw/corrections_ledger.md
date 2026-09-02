boss YC

# CORRECTIONS LEDGER — Open-MOPD 架构图复核

裁定基础：48 条判决（37 CONFIRMED / 4 IMPRECISE / 1 WRONG，其余 CONFIRMED 带注记）。我对 12 条做了独立抽查（含重跑 Hydra 组合探针），下面标注 `[已复核]` 的条目是我自己读源码/执行确认过的。

## 1. 已确认的承重事实

### 1.1 复现直接受阻的四道门（按触发顺序）

| # | 事实 | 位置 |
|---|---|---|
| G1 | 两个 PPO 启动脚本用**无 `+` 前缀**的 `actor_rollout_ref.rollout.reward_mode=...`，而该键在任何 yaml 与 `RolloutConfig` 里都不存在 → Hydra 组合期直接抛 `ConfigCompositionException`，Ray 还没起就死。修法就是加 `+`。`[已复核：重跑 /data/ycfeng/tmp/hydra_probe/probe.py，plain 形式 RAISED，`+` 形式 COMPOSED OK]` | `scripts/local/mt_opd.sh:87`、`scripts/local/opd.sh:66`；消费点 `training/verl/verl/trainer/ppo/ray_trainer.py:86-92`（`_pop_direct_opd_rollout_options` 读完即 `del`，正说明该键必须靠 `+` 注入） |
| G2 | 没有任何脚本/yaml 设 `log_prob_top_k`；trainer 读的是原始 DictConfig，`get(..., 0)` 拿到 0，`mt_opd`/`delta_opd` 第一步就 raise。dataclass 默认 256 是诱饵（worker 侧才生效）。`[已复核：探针输出 rollout.get('log_prob_top_k',0) -> 0]` | `ray_trainer.py:1968`、`:1973-1974`；`workers/config/rollout.py:154` |
| G3 | teacher 0 与 teacher 1..N-1 配置不对称：`input_tokenizer=null`、`use_remove_padding=True`、`param_offload=True` 只给 i≥1。teacher 0 沿用 yaml 默认 `input_tokenizer: ${actor_rollout_ref.model.path}`（非 null）→ `_do_switch_chat_template=True` → 索引 `non_tensor_batch["raw_prompt"]`，而 `return_raw_chat` 默认 False，该键不存在。`[已复核]` | `scripts/local/mt_opd.sh:90-91` vs `:107-111`；`trainer/config/reward_model/reward_model.yaml:33`；`workers/fsdp_workers.py:1810-1814`、`:2513/:2519`；`config/data/legacy_data.yaml:59` |
| G4 | `install_requirements.sh` 不装 vllm、不装 flash-attn（都在未被请求的 extras 里），而三个 PPO 启动脚本硬编码 `rollout.name=vllm`。`[已复核]` | `training/install_requirements.sh:10-23`；`training/verl/setup.py:50/:52`；`mt_opd.sh:86`、`opd.sh:65`、`rl.sh:55` |

### 1.2 MT-OPD 算法链路：发布态实际跑的是什么

| # | 事实 | 位置 |
|---|---|---|
| A1 | 发布版 `mt_opd.sh` 只给出 `+mt_opd.teacher_domains` 和 `+mt_opd.n_additional_teachers`，M1/M2/M3 全部落到默认关闭 → 纯 routing 版 M-OPD，`compute_domain_loss_weights` 返回 `None` | `mt_opd.sh:92-93`；`ray_trainer.py:664/674/706-708/746-748`；`workers/actor/mt_opd.py:207-208` |
| A2 | M2 的 `reward_scale_direction` 默认值就是源码自陈"实测有害"的 `divide`（注释记录 IF loss weight 27→91、梯度 token 占比 57% vs 目标 33%）；且 `divide` 分支无夹取，`multiply` 才夹到 [0.05, 20] | `mt_opd.py:166`、`:290-296`、`:316-325` |
| A3 | M3 的 [B,T] token 权重在调用点被压成 per-seq 后**直接乘进** `dom_w`，破坏了 kernel 精心维持的 token-加权均值=1 性质，且未重新归一。`[已复核]` | `ray_trainer.py:2216-2222`；性质来源 `mt_opd.py:332-335` |
| A4 | 冲突指标调用**不传** `conflict_nats`，阈值被钉在 kernel 默认 1.0，与 `mt_opd.conflict_nats` 脱钩（后者只进 policy）。`[已复核]` | `ray_trainer.py:2177-2183` vs `:2145`；`mt_opd.py:498`、`:538` |
| A5 | M4（`refresh_opd_advantage`）在 `RayPPOTrainer.fit` 上**不可达**：`teacher_on_student_log_probs` 在 `update_actor` 之前被无条件 pop。`[已复核：pop 块在 3478-3489，update_actor 在 3527，且是全仓唯一调用点]` | `ray_trainer.py:3479-3489`、`:3527`；gate `workers/actor/dp_actor.py:985`、`:1095-1099` |
| A6 | `domain_loss_weight` 乘在 **advantages** 上而非 `response_mask` 上（rank-generic `view`），全仓无其他施加点 | `dp_actor.py:1128-1132` |
| A7 | `build_domain_weights` 对未知 domain 不报错，给 1/N 均匀行；`compute_domain_loss_weights` 对 `target_shares` 里缺失的 domain 给 target 0.0 → 该 domain 序列梯度为 0（仅当**所有** present domain 都缺失时才回退均分） | `mt_opd.py:57-61`；`:225-227`、`:288`、`:328-334` |
| A8 | `conflict_policy="none"` 返回**同一个** tensor 对象（非拷贝）；`compute_teacher_conflict_metrics` teacher<2 时返回空 dict | `mt_opd.py:440-441`；`:525-526` |
| A9 | 非 mt_opd 的 else 分支设了 11 个 `mt_*` 属性，独缺 `mt_reward_scale_stat`（潜伏缺陷，因该属性只在 mt_opd 分支内被读）。`[已复核]` | `ray_trainer.py:766-777` vs `:718`、`:734`、`:2208` |

### 1.3 崩溃/死代码路径

| # | 事实 | 位置 |
|---|---|---|
| B1 | `top_k_strategy="top_p_intersec"` 在 `compute_distillation_reward` 里无分支也无 else → `rm_scores` 未绑定，`UnboundLocalError` 于 **773** 行；配置里却公开声明了该取值。兄弟函数 `compute_delta_opd_reward` 反而处理了它。`[已复核]` | `dp_actor.py:710/716/723/730/751`、`:773`；`workers/config/rollout.py:155` |
| B2 | `token_grpo` 注册了但不可达：签名无 `**kwargs`，而调用点必然转发 `true_reward_score`（fit 循环无条件写入）与 `responses` → TypeError。全仓无任何配置选用它。`[已复核]` | `core_algos.py:352-360`；`ray_trainer.py:548-561`、`:2486-2487` |
| B3 | `checkpoint.save_contents` 含 `hf_model` 对 SFT 阶段无效：调用方误用 `checkpoint_contents=`，被 `**kwargs` 吞掉，`checkpoint_config` 落为 None → 走默认三项，hf 权重分支永不执行。全仓仅此一处误用。`[已复核：grep `checkpoint_contents=` 只有一条命中]` | `workers/engine/fsdp/transformer_impl.py:164`；`utils/checkpoint/fsdp_checkpoint_manager.py:75-96`、`:308`；`checkpoint_manager.py:52-57` |
| B4 | SFT 的 `messages_key`/`tools_key`/`enable_thinking_key` 在 `sft_trainer_engine.yaml` 里是 `data:` 平铺键，而 dataset 只读 `config["multiturn"][...]`（该文件无 `multiturn:` 块）→ 三个键全是死配置，实际列名恒为硬默认 `"messages"` | `trainer/config/sft_trainer_engine.yaml:24-27`；`utils/dataset/multiturn_sft_dataset.py:62-66`、`:117` |
| B5 | `DomainWeightedSampler` 硬编码 `("math","code","if")`、无 `domain` 列即 raise，且**零生产调用方**（trainer 只 import 同模块的 `weighted_quotas`）；`data.sampler.class_path` 两处声明均为 null。`[已复核]` | `utils/dataset/domain_weighted_sampler.py:15`、`:57-59`；`ray_trainer.py:1797`；`config/data/legacy_data.yaml:122` |
| B6 | Megatron 后端没有 `compute_distillation_reward`（`grep distillation megatron_workers.py` = 0 命中），其 `compute_rm_score` 只转发 `self.rm.compute_reward`，无 top-k/teacher-logprob 支持 → 任何 OPD 模式在 `strategy=megatron` 下不可用。`[已复核]` | `workers/fsdp_workers.py:1091-1093` vs `workers/megatron_workers.py:1268-1275` |
| B7 | `evals/README.md` 记录的 5 个路径在发布物里全部不存在；`manage_third_party.py` 连 `--help` 都抛 `FileNotFoundError`（第 103 行在 `parse_args()` 之前就读 lock 文件）。`[已复核：5/5 MISSING]` | `evals/README.md:32-33/64/76/84-85/107/142/147-148`；`evals/verifier/scripts/manage_third_party.py:103` vs `:107` |

### 1.4 数据契约与 reward 语义

| # | 事实 | 位置 |
|---|---|---|
| D1 | 任意 parquet 列（含 `domain`）到达 trainer 的机制：`__getitem__` 原样返回整行 → `collate_fn` 把非 tensor 值塞进 object 数组 → `DataProto.from_single_dict` 分流到 `non_tensor_batch` | `utils/dataset/rl_dataset.py:338`、`:497`、`:83-111`；`protocol.py:492-500` |
| D2 | `_get_gen_batch` 的 keep-set 含 `"domain"`，因此 domain 留在训练 batch 上、sync 模式下**不进** gen_batch（async 模式才整体 update） | `ray_trainer.py:1149-1151`、`:1155-1159`、`:1162-1163` |
| D3 | `naive.py` 把 `rm_scores` 原样当训练 reward，规则分数只作为 `true_reward_score` 归档；`reward_extra_info` 在 279 行被**重新绑定**，所有逐行规则指标被丢弃 | `workers/reward_manager/naive.py:274-287` |
| D4 | `opd_val_dispatch` 的路由是**两个谓词 + 兜底**：`_is_if` → instruction_following(official_eval)，`_is_code` → rllm_code_reward(official_lcb)，其余（含 math）→ ttrl_math；输出被压成 `{score, acc}` | `utils/reward_score/opd_val_dispatch.py:27-36`、`:48-64`、`:67-87` |
| D5 | `evals/score_rollouts.py` **原地重写**源 parquet（新增 `score` 列），两个 code grader 把所有异常吞成 0.0 | `evals/score_rollouts.py:218`、`:223-226`、`:106-107`、`:140-141` |
| D6 | checkpoint 布局：SFT 在 `global_step_N/` 下直接放分片 + `fsdp_config.json` + `huggingface/`（仅 config/tokenizer，无权重）+ `data_{dp_rank}.pt`；PPO 深一层 `global_step_N/actor/` 且只有单个 `data.pt`。SFT→OPD 的必需转换只由 `training/scripts/sft/merge_model.sh` 描述，两个 README 均未提及 | `fsdp_checkpoint_manager.py:225-227/:258/:296`；`checkpoint_handler.py:70/:86-89/:98`；`ray_trainer.py:1515-1521/:1548`；`training/scripts/sft/merge_model.sh` |
| D7 | `actor.yaml` 的 `kl_loss_coef: 2.5` 与 `adaptive_kl_loss_max_coef: 2.5` 相等 → 自适应控制器起点即上限，只能单向下降；不过 `use_kl_loss` 与 `adaptive_kl_loss_coef` 默认都是 false，该条件要两者都打开才咬人 | `trainer/config/actor/actor.yaml:78/85/88/92`；`ray_trainer.py:3505-3520`；`kl_controller.py:8-14` |

## 2. 修正

### 2.1 会影响复现动作的（优先）

**C48 — 依赖安装（IMPRECISE）**
- 图说：`install_requirements.sh` 既不装 torch，也不装 vllm、flash-attn。
- 实际：vllm / flash-attn / numpy<2.0.0 / 三脚本硬编码 vllm 全部成立；但 torch **会**被 `pip install --user -e .` 传递安装——`install_requires` 里的 `accelerate`、`peft`、`torchdata`、`tensordict` 都依赖 torch。真实缺陷是"torch 版本与 CUDA build 完全不受控（走默认 index）+ 零 rollout 后端"，不是"没有 torch"。
- 证据：`training/verl/setup.py:26-45`（`accelerate`/`peft`/`torchdata`/`tensordict`）；`install_requirements.sh:13`。`[已复核]`

**C34 — `domain` 列的产出方（WRONG）**
- 图说：`experiments/data/build_mix_rl_raw_union.py` 是仓内**唯一**产出顶层 `domain` 列的 builder。
- 实际：至少还有 4 个。`experiments/data/build_mix_rl_union.py:108` `record["domain"] = domain`（并在 `:139/:147/:186` 当列读回，`:181` 落盘）；`training/scripts/sft/build_weighted_mix.py:15/:104/:138`（Arrow schema 里就有 `domain`）；`training/scripts/sft/shard_sft_parquet.py:15/:68/:100`；`experiments/data/rft_from_rollouts.py:132`（`:155` 按该列排序）。
- 成立的那一半：`training/scripts/rl/` 下**没有任何**脚本提到 `domain`（我实测该目录 8 个 `.py` 全部 0 命中——顺带修正 skeptic 说的"six files"）。所以准确说法是"MixRL 系两个 builder 里，它是带显式 Arrow `OUTPUT_SCHEMA` 字段的那个；单域 RL builder 一律不带 domain"。
- 证据：上述行号，`[已复核]`。

**C23 — Megatron 的失败点（IMPRECISE）**
- 图说：`compute_distillation_reward` 只在 FSDP worker 注册，Megatron 跑任何 OPD 都会**在 `ray_trainer.py:2152` 失败**。
- 实际：注册与缺失都对，但失败行随 `reward_mode` 变：exopd → `:2007`，mt_opd → `:2152`，标准 OPD(`top_k>0`) → `:2237`。而 mt_opd 下 Megatron 实际死得更早——`:2106` 取 `extra_raw.batch["teacher_on_student_log_probs"]` 时 KeyError，因为 megatron 的 `compute_rm_score`（`megatron_workers.py:1268-1275`）只返回原味 reward。
- 证据：`[已复核 megatron_workers.py:1265-1275 与 grep 0 命中]`。

**C43 — `param_merge.py` 的 CLI 契约（IMPRECISE）**
- 图说：README 写 `--base/--model/--output`，真实 flags 是 `--mode/--input NAME=PATH/--reference/--output-dir`。
- 实际：README 那条命令确实跑不了（`--model`、`--output` 不存在；还漏了两个 required 的 `--mode`、`--reference`），但 `--base` **是**真实 flag（`param_merge.py:296`，可选 `type=Path`，`:325` 消费），不该被列进"不存在"清单。另有 README 未提的 `--alpha`、`--alpha-for`、`--overwrite`。
- 证据：`[已复核 experiments/backend/param_merge.py:290-309 与 README.md:10-14]`。

### 2.2 读图精度修正（不阻塞运行）

**C24 — `token_reward_direct` 的 mask 广播（IMPRECISE）**
- 图说：函数体是 `advantages = token_level_rewards * response_mask.unsqueeze(-1)`。
- 实际：`.unsqueeze(-1)` 只在 `if token_level_rewards.dim() == 3:` 内部对 `response_mask` 施加（仅 dense top-k 情形），乘法本身是 `token_level_rewards * response_mask`；整段包在 `with torch.no_grad()` 里。注册行 `:876` 与 `returns = advantages.clone()` 精确。
- 证据：`core_algos.py:895-902`。`[已复核]`

**行号/范围偏差（CONFIRMED 判决内的自陈修正，逐条采信）**
- `compute_rm_score` 起点是 `fsdp_workers.py:2664`，不是 2662（C20）。
- `top_p_intersec` 的 `UnboundLocalError` 落在 `dp_actor.py:773`，2772 是空行（C17）。`[已复核]`
- `__init__` 的 mt_opd 校验块实为 `660-765`，`766-777` 是 else 默认赋值段；另有一条校验在范围之上的 `657-658`（C12）。
- `collate_fn` 的 `def` 在 `rl_dataset.py:83`，被引的 `96-111` 只是函数体（C35）。
- `apply_teacher_conflict_policy` 的调用在 `ray_trainer.py:2140-2147` 结束，`2148` 才是 `metrics.update(_cm)`（C8 子步 d）。

**范围被说窄或说宽的**
- C38 说"三个谓词"，实为**两个**谓词函数 + else 兜底（三条路由分支）。
- C40 的 unscored 行不只缺 `metric`：`official_extra` 与 6 个 behavior 键（trunc/repeat 系列）也一并缺失，raggedness 比图上写的更宽。
- C41 的空 outputs 行经 `pd.DataFrame.from_records` 会被兄弟行回填成 NaN 列，真正 ragged 只在"全空"时出现；另有条件戳入的 `stop_token_ids` 与 fallback 文件名 `{stem}_rollouts_*`。
- C21 的 `torch.gather` 只对 `{only_stu, union, union-intersection, top_p_intersec}` 生效；`intersection` 走的是对 teacher top-k 的 masked max（`fsdp_workers.py:1997-2009`），student token 落在 teacher top-k 外时得 `-inf`。
- C33 的"唯一"需限定：vendored verl 的 recipe/model docs 也泛泛提到 model_merger；就 Open-MOPD 的 SFT→OPD 交接而言 `merge_model.sh` 才是唯一在仓内说明该转换的产物。
- C3 的 kernel docstring（`mt_opd.py:188-189`，"Missing domains default to an equal split"）与代码不符，代码给的是 0.0；文档本身就是个待修的错。
- C1 的 `__all__` 漏掉 `refresh_opd_advantage` 无实际后果，`dp_actor.py:46` 显式 import。
- C25 的 `compute_token_reward_direct_plus_grpo_advantage` 返回注解仍写 `tuple[Tensor, Tensor]` 但实返 3 元组，靠调用侧 `len(res)==3` 兜住。
- C27 的 3-D 分支里有一条未加保护的 per-micro-batch `print`（`core_algos.py:1215`）。

## 3. 需要执行才能定论

本轮 48 条判决中 **UNVERIFIABLE_STATICALLY 为 0**——两条最关键的运行期结论（G1 Hydra 组合失败、G2 `log_prob_top_k`=0）已由探针实测，我也重跑复核过。下列条目静态证据链已闭合、但仍属"推断出的运行时行为"，如要签收建议各跑一条命令：

| 待定项 | 定论方式 |
|---|---|
| G1/G2 的失败顺序（先 Hydra 挂，加 `+` 后才撞 1974 行 raise） | `PYTHONPATH=/data/ycfeng/tmp/hydra_probe/site python /data/ycfeng/tmp/hydra_probe/probe.py`（已跑，输出如上）；再以 `+actor_rollout_ref.rollout.reward_mode=mt_opd` 起最小 `mt_opd.sh`，确认 traceback 落在 `ray_trainer.py:1974` |
| A5：M4 在 `fit` 上真不可达 | 在 `dp_actor.py:1095` 前插一行临时 print，或直接在 wandb/swanlab 里检查 `mt_opd/m4_advantage_refreshed`（`dp_actor.py:1108`）是否**从未**出现 |
| B1：`top_p_intersec` 崩溃 | `+actor_rollout_ref.rollout.top_k_strategy=top_p_intersec` + `+...log_prob_top_k=256` 起一步 OPD，看是否 `UnboundLocalError` at `dp_actor.py:773` |
| B2：`token_grpo` TypeError | `algorithm.adv_estimator=token_grpo` 跑一步，确认 `adv_estimator_fn(**adv_kwargs)` 抛 `unexpected keyword argument 'true_reward_score'` |
| B3：`hf_model` 不落盘 | `bash scripts/local/sft.sh` 存一个 step，`ls global_step_*/huggingface/` 确认只有 config/tokenizer/generation_config，无 `*.safetensors` |
| G3：teacher 0 的 `raw_prompt` KeyError | 起 `mt_opd.sh`（补齐 G1/G2）后看首个 `compute_rm_score` 是否 KeyError；对照实验：加 `reward_model.model.input_tokenizer=null` 后是否消失 |
| A2：`divide` 模式的实测有害性 | 源码注释引用的是 2026-08-04/06 的实验，仓内无对应 artifact；要复现需开 `+mt_opd.normalize_reward_scale` + `+mt_opd.target_gradient_shares`，跟踪 `mt_opd/domain/{dom}/loss_weight` 与梯度 token 占比 |
| G4：环境可用性 | `python -c "import torch, vllm, flash_attn"` 在 `install_requirements.sh` 之后执行，确认 vllm/flash_attn ImportError 而 torch 可导入 |

## 4. 净评估

作为**阅读指引**这张架构图是可信的：48 条里 43 条实质无误，唯一的 WRONG（C34）和 4 条 IMPRECISE 都是"范围说过头"或"细节记错"，没有一条把控制流讲反；行号绝大多数精确到行，偏差都在 ±2 内且已在本 ledger 逐条校正。可以直接拿它当 mt_opd kernel（`mt_opd.py`）、trainer reward block（`ray_trainer.py:2100-2230`）和数据契约链路（parquet→`non_tensor_batch`）的导航图。需要最大警惕的是三块：**（一）"唯一 / 从不 / 全部"这类全称判断**——C34 就是在这里翻车，凡涉及"仓内唯一产出方/唯一调用方"的断言都应自己 grep 一遍；**（二）跨后端与 CLI 契约描述**——Megatron 失败点（C23）、`param_merge` flags（C43）、`install_requirements` 的 torch（C48）三处失准都集中在"图作者没实际跑过的边缘路径"；**（三）发布态与设计态的落差**——图对 M1/M2/M3/M4 的描述是设计意图，而 shipped `mt_opd.sh` 只启用了 routing，且 M4 在主训练路径上完全不可达（A1/A5）；读图时必须把"kernel 实现了什么"和"发布脚本会走到什么"分开看，否则会照着图去调一批根本没生效的旋钮。