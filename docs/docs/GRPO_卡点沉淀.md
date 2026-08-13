# GRPO 卡点沉淀：4B LoRA OOM / 多卡适配 / verl 与用户代码边界

> 版本：2026-08-10。沉淀对象：4×4090 24GB + veRL 0.8.0 + Qwen3-4B + SFT LoRA 的正式 Agentic GRPO。
> 适用：后续任何涉及多卡 veRL LoRA GRPO / SFT 的任务先读本文档。

---

## 一、Codex 一直没搞定的 OOM：根因、解决思路、关键代码位置

### 1.1 现象（Codex 6 次预检 + 我方 1 次，数字完全一致）

- 4×4090 24GB 跑 Qwen3-4B + SFT LoRA（step_136）GRPO，**veRL 0.8.0 默认配置必在第一次 `update_weights` OOM**：FSDP 20.95GB + vLLM 2.53GB = 23.48/23.52GB。
- **TP=1 / TP=2 / TP=4 全部在同一位置失败**（反分片是全模型操作，与 TP 无关）。
- 反复试参（压 batch、缩上下文、调 gpu_memory_utilization 0.2→0.35→0.4、TP=2×2）最多把差距从 ~300MB 压到 ~76MB，**从未突破**——因为方向错了。

### 1.2 错误归因 vs 正确归因

| 阶段 | 归因 | 结论 |
|---|---|---|
| 前期（错） | "显存差一点，压 batch / 缩上下文 / 调 KV 配额" | **不是计算/显存配置问题**。batch 与上下文只影响 rollout 阶段，OOM 恒发生在权重同步路径 |
| 后期（错） | "再加一张卡" | 反分片峰值与卡数无关（gather 的是完整基座 8GB），加卡解决不了 |
| 正确 | **LoRA 基座同步路径**：veRL 在 update_weights 时把完整基座反分片到单卡 | 见 1.3 根因链 |

### 1.3 根因链（veRL 0.8.0 源码，只读定位）

```
engine_workers.py:614   base_sync_done = "dummy" not in config.rollout.load_format
                        默认 load_format=dummy → base_sync_done = False
engine_workers.py:711   update_weights():
                          do_lora_base_sync = not base_sync_done  → True
                          → get_per_tensor_param(base_sync_done=False)
                          → FSDP.summon_full_params()   ← 把完整基座 8GB 反分片到 GPU
```

**第二个坑（只修 load_format 不够）**：即使 `base_sync_done=True`，`update_weights` 里
`summon_full_params(module)` **仍然会 gather 完整模型**，只是事后用
`get_peft_model_state_dict` 过滤出 LoRA keys —— 峰值内存一样爆。所以必须让
gather 本身逐层进行。

### 1.4 解决思路（零代码补丁，纯配置）

1. **`actor_rollout_ref.rollout.load_format=safetensors`** → vLLM 自载基座权重 →
   `base_sync_done=True`，跳过 `do_lora_base_sync` 的全量反分片。
2. **`actor_rollout_ref.rollout.layered_summon=true`** → `collect_lora_params` 走
   `layered_summon_lora_params`（fsdp_utils.py:673），逐层 summon LoRA 参数，
   峰值 = 单个 transformer layer ~0.2GB，而非完整基座 8GB。
   （veRL 原生支持；报错信息明确要求 base_sync_done=True 才能启用。）

### 1.5 关键代码位置

| 位置 | 内容 |
|---|---|
| `engine_workers.py:614` | `base_sync_done` 判定（load_format 是否 dummy） |
| `engine_workers.py:700-745` | `update_weights` 全流程（do_lora_base_sync → get_per_tensor_param） |
| `utils/fsdp_utils.py:620-740` | `collect_lora_params` / `layered_summon_lora_params`（逐层 gather 实现） |
| `vllm_async_server.py` | HYBRID 模式下 load_format 原样透传给 vLLM |
| `scripts/run_grpo_agentic.sh` | 我方入口：`LOAD_FORMAT` / `LAYERED_SUMMON` 环境变量 → Hydra override（空值 = verl 默认） |

### 1.6 验证结果（2026-08-10 实测）

- toy 预检：FSDP 峰值 **20.95GB → 6.82GB** allocated / 8.19GB reserved，vLLM ~3GB，余 8.5GB headroom。
- 正式池 5-step：FSDP 峰值 11.47GB allocated / 14.89GB reserved，`update_weights` 首步 8.1s、后续 2-4s。
- 24GB 单卡全程无 OOM；clean exit。

### 1.7 配套约束（同一卡点群，缺一不可）

| 约束 | 原因 |
|---|---|
| torchrun **单 driver**：`NPROC_PER_NODE=1` + EXTRA_ARGS `trainer.n_gpus_per_node=4` | 4 个 torchrun rank 各自拉起 ray 集群 → OpenBLAS 线程风暴 → `pthread_create failed thread 55 of 64: Resource temporarily unavailable`，整群 worker 死亡 |
| 线程配额 OMP/OPENBLAS/MKL/NUMEXPR=8 | sweforge env 每进程默认 64 线程，多进程叠加必 EAGAIN |
| **禁止** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | veRL/vLLM 的 CuMem 池不兼容（SFT 可用，GRPO 必须 unset） |
| TP=4 + gpu_memory_utilization=0.2 | 每卡权重仅 2GB，峰值 ~15.5GB/24GB；TP=1 需 u≥0.4（权重 8GB 常驻）更紧 |

### 1.8 经验沉淀（本项目卡点模式）

1. **veRL LoRA GRPO 的 OOM 先查权重同步路径，不要压 batch**。`update_weights` / `state_dict` / `summon` 关键字命中即本卡点。
2. veRL 默认 `load_format=dummy` 与 FSDP LoRA 是组合坑：dummy 意味着"基座由 FSDP 反分片供给"，而 vLLM 其实可以自载。
3. 反分片峰值与卡数无关 → "加卡"不是万能；配置修复前不要盲目加资源。
4. 多进程框架（ray/torchrun/vLLM worker）在容器环境优先压线程配额，再怀疑显存。
5. 涉及 LoRA 的 load_format 修改后，必须验证 `base_sync_done` 语义（config 打印或源码断点），配置生效与否看日志中的 update_weights 耗时（~8s 首步 vs OOM）。

---

## 二、本项目多卡 GRPO 的"多卡适配"设置

（vs 普通单卡/朴素多卡跑法；全部经 4 卡实测）

| 设置 | 值 | 多卡意义 |
|---|---|---|
| 训练驱动 | torchrun `--nproc_per_node=1` + `trainer.n_gpus_per_node=4`（EXTRA_ARGS 覆盖） | 4 卡归**一个** ray/verl 集群；4 份 driver = 4 个集群 = 线程风暴（§1.7） |
| rollout 并行 | vLLM `tensor_model_parallel_size=4`（TP=4, 单副本） | 无 NVLink 的机器 TP 优于 DP：每卡只驻留 1/4 权重（2GB）+ 1/4 KV；DP/多副本会让每卡各驻一份完整权重 |
| 显存配额 | `gpu_memory_utilization=0.2` | 只给 vLLM 2GB 预算（4B 权重已由 TP 分摊），把 ~21GB 留给 FSDP 训练侧 |
| 权重同步 | `load_format=safetensors` + `layered_summon=true` | vLLM 自载基座（base_sync_done=True）+ 逐层 gather → 多卡 FSDP 侧峰值最小化（§1.4） |
| 单卡 token 预算 | `ppo_max_token_len_per_gpu` = prompt+response（脚本自动算） | 只设 response 会触发 verl AssertionError（阶段 18 E2E 实测） |
| KV cache 上界 | `max_model_len=4096` 显式 | 防止默认 32768 的 KV 预留超出 u=0.2 的 2GB 预算（阶段 17 实测 3.5GiB > 3.07GiB） |
| 调度预算 | `max_num_seqs=4` + `max_num_batched_tokens=4096` | 每 step 4 任务 × n=4 = 16 序列，分批调度不超 4096 token/趟 |
| 训练侧 | FSDP（veRL HYBRID：FSDP TrainingWorker 与 vLLM 同进程同卡） | 4B 权重 + LoRA + 优化器状态分片在 4 卡，CPU offload 兜底（峰值 127GB RAM） |
| 线程 | OMP/OPENBLAS/MKL/NUMEXPR=8 | 多 worker 进程 × 每进程 64 线程 = EAGAIN（§1.7） |
| 实测峰值 | FSDP 11.5GB + vLLM ~3GB ≈ 15GB/24GB | 余 ~9GB headroom；TP=4 布局下 4B 可稳定运行 |

核心思想：**24GB 装不下 4B 完整权重时，让每一层（FSDP 分片、TP 分权重、vLLM 自载、逐层 gather）都只碰它该碰的那一小块**，峰值由"最大单次反分片"决定，而不是"模型总大小"。

---

## 三、GRPO 中：verl 决定的部分 vs 用户写的部分

### 3.1 verl（框架层，版本锁定 0.8.0，只读）决定

- **训练主循环编排**：`main_ppo` / RayPPOTrainer 的 rollout → compute_score → advantage → update_actor → update_weights 调度、global_step 推进、train/val 流程。
- **FSDP 分片与梯度同步**：`actor_rollout_ref.actor.fsdp_config.*`（use_torch_compile 等）、ZeRO 状态管理。
- **vLLM rollout 引擎**：`actor_rollout_ref.rollout.*`（name=vllm、TP、KV cache、max_num_seqs、batch 调度、load_format 语义）。
- **GRPO 算法**：`algorithm.adv_estimator=grpo`、组内归一化、`use_kl_in_reward` / `use_kl_loss`、ppo_epochs 循环、pg_loss/clip。
- **DataProto / dataloader / mask 基线装配**：prompt/response 打包、old_log_prob 计算、rm_scores 与 last-valid-token 的放置语义。
- **checkpoint 引擎**：naive backend、save/load、resume。

### 3.2 用户写的（本项目 `src/` + `scripts/` + `configs/`）

- **Agent Loop**：`src/sweforge/rollout/verl_agent_loop.py`（SWEForgeVerifierLoop）——多轮生成-执行-观测循环、终止判定、extra_fields 落盘；经 `configs/agent_loops.yaml` 注册为 verl 的 `default_agent_loop`。
- **环境与工具协议**：canonical 五工具（bash/search/view_file/str_replace/finish）+ parse + render_observation（协议层，SFT 与 RL 共用）。
- **EnvironmentBackend**：`src/sweforge/environment/*`（Remote 客户端 + §8 契约 server + Mac Docker 执行 + 唯一 env instance 生命周期）。
- **Reward**：`src/sweforge/reward/verifier.py`（CleanVerifier，git apply + build + test 真实执行）+ `schemas/verification.py` 的 `binary_reward`（纯二进制 1.0/0.0）+ **`partial_reward`（2026-08-11 v4: F2P partial 主导, resolved 锚 1.0, 见 GRPO_正式运行手册.md §6）**——**reward 数值是用户的，advantage 计算是 verl 的**。
- **数据管线**：`src/sweforge/data/rl/prepare_grpo.py`（TaskSpec → RL jsonl、policy_view 剥离 secret、seed 确定性）。
- **模型接入**：SFT LoRA 导出（export_lora）+ `lora_adapter_path` 语义 + vLLM 双验证。
- **全部参数化入口**：`scripts/run_grpo_agentic.sh`（多卡布局、显存、线程、save 策略、健康检查硬 gate）。
- **可观测性**：extra_fields 全键、`observability/` 模块（曲线/系统采样）。

### 3.3 边界示例（判断归属）

| 事件 | 归属 |
|---|---|
| `update_weights` 执行 | verl（engine_workers.py） |
| 它 OOM | 我们的配置修复（load_format + layered_summon） |
| reward_score（v4: F2P partial; 默认 0.05~0.95, resolved=1.0） | 我们（CleanVerifier + partial_reward） |
| 它放进 last valid token / 组内归一化 | verl |
| 多轮 rollout 生成与工具调用 | 我们（AgentLoop + Backend） |
| prompt 打包 / mask / logprob | verl 装配，但 token 纪律（response_ids 来自 vLLM 不重 tokenize）由我们保证 |
| checkpoints 保存 | verl（naive backend），保存路径/频率由我们配置 |

**一句话**：verl 决定"怎么训练"（循环、分片、算法、引擎），我们决定"训练什么"（数据、任务、环境、奖励、多卡配置）。改动 verl 是禁区（铁律 6），一切适配走配置与注册接口。
