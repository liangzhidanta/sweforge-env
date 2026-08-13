# SWE-Forge 正式 GRPO 运行手册

> 版本：2026-08-11 v4
> - v4（2026-08-11）：**奖励改为 F2P partial 主导**（§6, `partial_reward`, 2026-08-11
>   v2 规格）——链路修复后模型能力层的 reward 恒 0（15/16 RESPONSE_CAP, 修不对）
>   暴露出**纯二进制在"全部 unresolved"的 step 里 advantage 恒 0, pg_loss 恒 0
>   空转**（模型学不到）。F2P partial: resolved → 1.0 锚不变; 未解决时按逐条
>   F2P 测试真实通过情况给部分分（+ P2P 破坏惩罚防刷分 + format 小台阶）,
>   每分来自 CleanVerifier 真实 exit code, 不伪造（P5）。extra_fields 加
>   `reward_score`（实际进训练的主键）与 `reward_breakdown`（F2P 明细）;
>   `verifier_reward` 保持纯二进制（解决率指标不污染）。
> - v3（2026-08-11）：**reward 恒 0 二阶段修复落地**（docs/GRPO_卡点沉淀_奖励恒0_SFT格式与rollout差距.md）：
>   harness 多调用消费（parse_all_tool_calls, 一轮全部调用执行, 见 §2 代码表）、
>   `--temperature 0.7`（1.0 采样发散为多语言乱码）、`--turn-budget 512`
>   （轮预算 256 内写不完叙述+完整 JSON）。§3.3 命令更新; 故障表补
>   hydra 裸键 EXTRA_ARGS 与 KV cache OOM（§7.4）。注意: 链路修复后
>   step 1 实测 reward 仍 0（15/16 RESPONSE_CAP 终止, 模型不会 finish/
>   修不对）——属模型能力层, GRPO 该学的, 与前"链路坏"本质不同。
>   该层由 v4 的 F2P partial 解决。
> - v2（2026-08-10 晚）：训练池换为 **Mixed 142 条**（Public 42 / AST 50 / Reversal 50，
>   `rl_mixed_v2/pool_v1`）；沉淀**单 driver 冒烟配方**（4B+step_136+pool_v1 实测通过，
>   见 §3.2 ①）；Mac bundle 查找加后缀剥离修复（§5）；故障表补
>   `invalid device ordinal`（多 driver 症状）与空工作区排查（§7.4）。
> - v1（2026-08-10）：预检 4B+step_136+4 卡闭环已通过，global_step:1。
> 适用范围：AutoDL 4×4090 24GB + Mac Docker CleanVerifier 的正式 4B LoRA GRPO。

---

## 1. 链路总览

```
prepare_grpo（TaskSpec pool → RL jsonl, policy_view 剥离 secret）
  → torchrun 单 driver → veRL main_ppo
  → SWEForgeVerifierLoop（configs/agent_loops.yaml 注册, 每并发 rollout 唯一 env instance）
     vLLM 生成 → canonical parse（五工具）→ RemoteEnvironmentBackend
     → SSH 隧道 → Mac Environment Server → LocalDocker 容器真实执行
     → export_patch → CleanVerifier（F2P/P2P/integrity）→ partial_reward
     （F2P partial, resolved 锚 1.0, 2026-08-11 v2）
  → reward_score 落最后有效 response token → GRPO advantage → update_actor
```

---

## 2. GRPO 相关代码文件

| 文件 | 职责 |
|---|---|
| `src/sweforge/rollout/verl_agent_loop.py` | **SWEForgeVerifierLoop**：多轮 AgentLoop（canonical parse + 五工具 + render_observation + finish/终止 → export_patch → CleanVerifier → `partial_reward` → `reward_score`）。extra_fields: reward_score（进训练的主键）/ verifier_reward（纯二进制）/ resolved / F2P / P2P / integrity_ok / patch_nonempty / reward_breakdown（F2P 明细）/ termination_reason / tool_call_count / parse_error_count / action_trace |
| `src/sweforge/data/rl/prepare_grpo.py` | TaskSpec 池 → veRL GRPO jsonl。`task.policy_view()` 剥离 secret（gold/hidden/mutation 结构上不可见，P5）；`seed=7` 确定性抽样；`--max-tasks` 截断。输出行 = `{data_source, prompt(canonical messages), extra_info:{task: policy_view, index}}` |
| `src/sweforge/reward/verifier.py` | CleanVerifier + Isolation（TempDirIsolation / DockerIsolation）：git apply + build + test 真实执行，不伪造 reward；`partial_reward`（§6） |
| `src/sweforge/schemas/verification.py` | `binary_reward()`（纯二进制, `verifier_reward` 键）、`is_resolved`、`VerificationResult`、F2P/P2P/integrity 语义 |
| `src/sweforge/environment/{base,mock,remote,server}.py` | EnvironmentBackend 接口 / Mock / Remote 客户端（8 端点 §5）/ §8 契约 server |
| `src/sweforge/environment/taskenv.py` | TaskEnvironmentBackend（TaskSpec → 模板 clone + checkout base + 注入 tests）+ GitRepoIsolation |
| `src/sweforge/training/sft/export_lora.py` | FSDP ckpt → PEFT adapter 导出（双验证） |
| `scripts/run_grpo_agentic.sh` | **正式训练入口**（§3 命令；支持 NPROC_PER_NODE / N_GPUS_PER_NODE / GPU_MEM_UTIL / LOAD_FORMAT / LAYERED_SUMMON 环境变量） |
| `scripts/verl_ppo_entry.py` | torchrun 入口 wrapper（import verl 前 pop `TORCHELASTIC_USE_AGENT_STORE`） |
| `configs/agent_loops.yaml` | AgentLoop 注册表：`sweforge_verifier`（env_url=http://127.0.0.1:8500, max_assistant_turns=12, constrained_smoke=false） |
| `configs/data_recipe.yaml` | 数据配方（§4） |
| `scripts/flash_attn_shim/` | transformers 4.57 fallback shim（verl 0.8.0 `_compute_old_log_prob` 硬 import） |
| `scripts/interop_mac.py` | Mac 联通冒烟（health + 五工具 + verify 7/7） |

**veRL 侧关键源码位置（锁版本 0.8.0，只读参考，勿改）**：
- `engine_workers.py:614` `base_sync_done = "dummy" not in rollout.load_format`
- `engine_workers.py:711` `update_weights`（`do_lora_base_sync = not base_sync_done`）
- `fsdp_utils.py:673` `collect_lora_params`（`layered_summon` 逐层 gather）
- `vllm_async_server.py`（HYBRID 时 `load_format` 原样透传）

---

## 3. 训练 Bash 命令

### 3.1 铁律前置（每次运行前）

```bash
unset PYTORCH_CUDA_ALLOC_CONF        # veRL/vLLM CuMem 池禁止 expandable_segments
export NPROC_PER_NODE=1              # torchrun 单 driver（严禁 4 份 PPO driver）
export N_GPUS_PER_NODE=1             # 4 卡由 EXTRA_ARGS 的 trainer.n_gpus_per_node=4 声明
export GPU_MEM_UTIL=0.20
export LOAD_FORMAT=safetensors       # vLLM 自载基座 → base_sync_done=True
export LAYERED_SUMMON=true           # 逐层 gather LoRA（消除 8GB 反分片峰值, 决定性）
# 线程限制已内置于脚本（OMP/OPENBLAS/MKL/NUMEXPR=8, 防 AutoDL 线程 EAGAIN）
```

### 3.2 三阶段命令

| 阶段 | 命令要点 | 通过标准 |
|---|---|---|
| **① 预检**（1 step, pool_v1 抽 2 任务 × n=4 = 8 rollout） | 实测配方（2026-08-10 v2, 4B+step_136）: `--pool .../rl_mixed_v2/pool_v1/train/pool.jsonl --steps 1 --batch 2 --n 4 --max-response-length 1024` + EXTRA_ARGS: `trainer.n_gpus_per_node=4`（单 driver; 严禁 NPROC=4, 见 §7.4） | 见 §7.5 |
| **② 稳定性**（5 step, 正式池抽一个 batch） | `--pool .../dataset_v1/train/pool.jsonl --steps 5 --batch 4 --n 4 --max-prompt-length 3072 --max-response-length 1024` + EXTRA_ARGS: `trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=4 actor_rollout_ref.rollout.max_model_len=4096 actor_rollout_ref.rollout.max_num_seqs=4 actor_rollout_ref.rollout.agent.num_workers=1 actor_rollout_ref.actor.optim.lr=5e-7 actor_rollout_ref.actor.ppo_epochs=1` | 无 OOM；reward 不长期全 0/全 1；advantage 有方差；loss/grad_norm 有限；每步 update_weights → rollout → verify → update_actor |
| **③ 正式**（50 step） | 同② + `--steps 50 --save 25`（保存 step25/50 两份） | resolve rate / KL / reward 决定是否续到 100 step |

### 3.3 完整正式命令（阶段 ③, v3 2026-08-11）

```bash
cd /root/autodl-tmp/SWE_project
unset PYTORCH_CUDA_ALLOC_CONF
export NPROC_PER_NODE=1 N_GPUS_PER_NODE=1 GPU_MEM_UTIL=0.20 \
       LOAD_FORMAT=safetensors LAYERED_SUMMON=true
# 产物落项目路径: 脚本默认 outputs/<date>/grpo-agentic-<time>/, 也可显式指定
export RUN_DIR=/root/autodl-tmp/SWE_project/outputs/2026-08-11/grpo_4b_formal_v3

bash scripts/run_grpo_agentic.sh \
  --model /root/autodl-tmp/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --adapter /root/autodl-tmp/experiments/sft_qwen3_4b_lora_r16_e1_20260809/eval/adapters/step_136 \
  --pool /root/autodl-tmp/experiments/rl_mixed_v2/pool_v1/train/pool.jsonl \
  --out-dir "$RUN_DIR" \
  --steps 50 --batch 4 --n 4 --save 25 \
  --max-prompt-length 3072 --max-response-length 2048 \
  --turn-budget 512 --temperature 0.7 \
  -- \
  trainer.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_epochs=1
```

参数说明：每 step 4 任务 × n=4 = 16 条 rollout；`ppo_max_token_len_per_gpu` 由脚本自动算为 prompt+response（4096，铁律）；save_freq=25 → step25/50 两份 ckpt。**`--no-save` 不是合法参数**（脚本只认 `--save N`；默认即不保存，save_freq=-1，传了直接 exit 2）。

v3 变更要点（2026-08-11 实测）：
- `--temperature 0.7`（默认已改）：1.0 时 4B 低概率区发散为泰文/阿拉伯文/希伯来文乱码（probe 实测 turn 2 全段）。
- `--turn-budget 512`：256 内模型写不完"叙述+完整 JSON"（默认 256 → JSON 截断 → PARSE_ERROR 提前终止）。
- `--max-response-length 2048`（v2 是 1024）：多轮总预算；实测 step1 response_length mean 2005。
- harness 多调用消费已内置（parse_all_tool_calls：一轮 2-3 个调用全部执行，SFT 训练目标一轮 1 个，模型自回归续写多调用被丢弃 = 动作损失）。
- **EXTRA_ARGS 必须是完整点路径**（hydra 裸键直接 `Could not override` 崩）：`max_model_len=4096` 裸键不行，要 `actor_rollout_ref.rollout.max_model_len=4096`；`tensor_model_parallel_size=4` 必带（缺它 vLLM 单卡载 8GB 权重 + u=0.2 → KV cache 无内存报错）。
- 实测 step1（2026-08-11）：num_turns mean 3.44（2-5 轮真实执行）、tool_calls 28s/rollout、entropy 0.625、564s/step（50 step ≈ 7.8h）；**reward 仍 0**（15/16 RESPONSE_CAP 终止）——链路已通，模型能力层待 GRPO 学。

> 全池覆盖扩展（未实测，默认不启用）：脚本默认 `prepare_grpo --max-tasks=BATCH`，即 50 step 反复训练同一批 4 任务（已验证路径）。若要覆盖更大池面：先 `prepare_grpo --pool <pool.jsonl> --out rl_full.jsonl --max-tasks 142`（train_batch_size=4 需整除，142/4=35.5 不整除 → 取 140 或 144），再用 `--train-jsonl rl_full.jsonl` 启动，每 step 从池中取不同 4 条。

---

## 4. 数据用量

### 4.1 任务池（当前实测）

| 池 | 路径 | 条数 | 构成 |
|---|---|---|---|
| **正式训练池（当前）** | `/root/autodl-tmp/experiments/rl_mixed_v2/pool_v1/train/pool.jsonl`（full 同目录, 含 gold/mutation 私有字段） | **142**（1.2MB） | **Mixed v1**: Public 42（aiohttp, r2e-gym 已验证）/ AST 50（humanize 17 + dateutil 15 + toolz 18, 六类变异）/ Reversal 50（more_itertools 34 + dateutil 7 + toolz 9, 真实 repair 反转）；manifest.json 记录配额 42/50/50 与比例（Public 29.6%, 50% 目标后补） |
| 旧正式池（已弃用） | `.../rl_formal_v1/dataset_v1/train/pool.jsonl` | 250 | 全 AST_MUTATION（无 public/reversal） |
| 评测池 | `.../rl_formal_v1/dataset_v1/eval/pool.jsonl` | 4（仅链路；正式结论前需扩到 20–50） | 仓库与训练隔离 |
| 联调池 | `.../sft_qwen3_4b_lora_r16_e1_20260809/parallel/rl_data_v1/pool_train.jsonl` | 10 | demo_calc + aiohttp（Mac Docker 可执行） |
| 预检池 | `/root/autodl-tmp/SWE_project/data/grpo_toy_pool.jsonl` | 4 | demo toy 任务 |

### 4.2 数据流（每 step）

```
pool.jsonl（142 任务, 完整 TaskSpec 含 gold/hidden/mutation）
  → prepare_grpo（policy_view() 剥离 secret + seed=7 抽样 + --max-tasks batch 截断）
  → rl_prompts_train.jsonl（{prompt: canonical system + [ISSUE], extra_info.task: policy_view}）
  → veRL DataLoader（train_batch_size=4）
```

- **secret 剥离验证**：policy_view 仅含 `task_id/repo/base_commit/problem_statement/environment`；gold patch / hidden test / mutation 结构上不可见（P5，`tests/test_prepare_grpo.py` 断言）。
- **每 step token 量（估算）**：4 任务 × 4 rollout ×（prompt ~3072 + 多轮 response ~1024×2-8 轮）≈ 每 step 10–30 万 token（toy 预检实测：8 条 rollout 3320 token，52s/step）。

### 4.3 数据配方（`configs/data_recipe.yaml`）

| 项 | 值 | 含义 |
|---|---|---|
| `rl_task_pool.source_weights` | ast_mutation 0.50 / repair_reversal 0.20 / public_executable 0.30 | 三来源采样权重（和=1.0），必须配置化 |
| `difficulty_profiling.G` | 2 | 每任务独立 rollout 次数 |
| `difficulty_profiling.thresholds` | easy ≥1.0 / unsolved ≤0.0 | resolve_rate 分档 |
| `priority_pool.base_priority` | p*(1-p) | 基础优先级，mixed 优先；source/repo/under_sampled 各 +0.2/+0.2/+0.1 |
| `flywheel.generator_controller` | upweight 1.5 / downweight 0.2 / target_band [0.3,0.8] / smoothing 0.5 | 有界权重更新 |

注：当前正式池 250 条全部来自 AST_MUTATION 来源；recipe 的 0.50/0.20/0.30 是飞轮目标的配置，正式规模数据（Repair Reversal / Public Executable）扩展属训练外层独立阶段（见《SWE-Forge_GRPO训练数据构造与分布说明.md》）。

---

## 5. Docker 使用情况

**AutoDL 侧零 Docker 依赖**（铁律）：Agent Loop 只走 `EnvironmentBackend` HTTP（8 端点 §8 协议），Mac 侧容器对 AutoDL 完全透明。

| 环节 | 实现 |
|---|---|
| Mac Environment Server | conda base 启动：`env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY NO_PROXY=127.0.0.1,localhost PYTHONPATH=src /opt/homebrew/Caskroom/miniconda/base/bin/python -m sweforge.env_server.server --bundles-dir bundles_v2 --docker --cleanup-stale 3600 --port 8501`（当前 bundles_v2 342 目录 = 250 AST/formal + 42 r2e-aiohttp + 50 reversal，142/142 pool 全覆盖） |
| **bundle 查找后缀剥离** | **Mac 侧 `env_server/docker/backend.py` `_bundle()`（约 168-184 行）已加补丁**：精确匹配 `bundles-dir/<task_id>` 失败时剥离尾部 `-[0-9a-f]{12}` 再查（AutoDL 并发唯一后缀）；两者皆无回落空模板。**无此补丁 = 每 rollout 空工作区 = reward 恒 0**（2026-08-10 真根因, 见 docs/GRPO_卡点沉淀_空工作区_task_id后缀.md）。补丁后 AutoDL 侧 142 全覆盖探针全命中 |
| 反向隧道 | Mac: `ssh -N -R 8500:127.0.0.1:8501 -i ~/.ssh/autodl_ed25519 -p <端口> root@<AutoDL 地址>` |
| 每 rollout 环境 | 每并发 rollout 唯一 env instance（uuid 后缀），`destroy` 幂等清理；`max_assistant_turns=12` |
| 工具执行 | bash / search / view_file / str_replace / finish 五工具在容器内真实执行（LocalDocker backend, docker=true） |
| 验证 | rollout finish/终止 → `export_patch` → CleanVerifier：容器内 `git apply` + build + 测试命令真实执行（F2P 必须 fail→pass，P2P 必须 pass→pass） |
| 端点 | `GET /health`；`POST /v1/tasks/register`、`/v1/envs`、`/v1/envs/{id}/reset`、`/v1/envs/{id}/actions`、`/v1/envs/{id}/patch`、`/v1/verifications`；`DELETE /v1/envs/{id}`（request_id 幂等） |

**联通检查**：`curl -s http://127.0.0.1:8500/health` → `{"ok":true}`；`PYTHONPATH=src python scripts/interop_mac.py --base-url http://127.0.0.1:8500` → 7/7。server 不可达时 `run_grpo_agentic.sh` 直接 exit 2，不允许 Mock 冒充。

---

## 6. 奖励函数计算公式

### 6.1 partial_reward（F2P partial 主导，2026-08-11 v2；P5 不伪造）

```
F2P_pass  = 全部 fail_to_pass 测试 passed（修复前必失败、修复后通过）
P2P_pass  = 全部 pass_to_pass 测试 passed（修复前后均通过）
integrity_ok = patch 干净应用（git apply 成功；空 patch 合法但由测试判 unresolved）

resolved  = verdict == "resolved" AND F2P_pass AND P2P_pass AND integrity_ok

reward(r) = 1.0                                    若 resolved（唯一满分锚）
            0.0                                    若 integrity_ok=False（apply 失败）
                                                   或 env_error / verification 缺失
            0.0                                    若 f2p 为空（异常数据, 不 vacuum）
            否则:
                f2p_credit = Σ w_i·passed_i / Σ w_i   （w_i 默认等权, 按 test_id;
                                                        v2 难度加权入口）
                p2p_factor = 1 − λ·(broken/total)    （λ=0.5; 防破坏性修复刷分）
                reward(r)  = f2p_credit·p2p_factor·0.95 + format_ok·0.05
                            （format_ok = 1 若 >=1 个合法工具调用, 弱模型 bootstrap）

未解决时 reward < 1.0 严格成立: 全 F2P 过 + 全 P2P 过 + integrity = resolved,
锚先接管; 否则必有 f2p_credit < 1 或 p2p_factor < 1。
```

- **为什么 partial**（2026-08-11 实测）: 纯二进制下"全部 unresolved"的 step
  advantage 恒 0 → pg_loss 恒 0 → 模型学不到（16/16 rollout reward 0）。
  F2P partial 把信号对准真目标（测试真的变绿）: 每分来自 CleanVerifier
  逐条 TestResult 的真实 exit code, 事实性不伪造（P5 铁律是"不伪造",
  不是"必须二进制"）。
- 每条 rollout 的 `reward_score` 由 CleanVerifier 真实执行得出，**落最后有效 response token**（rm_scores），进 GRPO advantage 计算。
- 环境/管线异常（Mac 不可达、env 创建失败等）→ 终止 + reward 0（不伪造、不让整步崩溃，`termination_reason=ENV_ERROR`）。
- 键位（可观测性，不进 policy）：`reward_score` = 实际进训练的主键（partial）;
  `verifier_reward` = **纯二进制**（resolved 1.0 / 其他 0.0, 解决率指标不污染）;
  `reward_breakdown` = {binary, f2p_passed/f2p_total, p2p_broken/p2p_total}
  （resolved 时为 {binary, resolved: 1.0}）; 其余: verifier_resolved /
  verifier_f2p_passed / verifier_p2p_passed / verifier_integrity_ok /
  patch_nonempty / termination_reason / tool_call_count / parse_error_count /
  action_trace。
- GRPO advantage：组内（同任务 4 条 rollout）归一化；reward 全同 → advantage 全 0 → pg_loss=0（模型/任务行为问题，非链路问题）。partial 正是为打破全同而设。

---

## 7. 操作教程

### 7.1 开机联通（每次）

```bash
# Mac: docker info 确认 Docker/Colima → 启动 Environment Server（§5）→ 建立反向隧道
# AutoDL:
curl -s -m 5 http://127.0.0.1:8500/health      # 期望 {"ok":true}
cd /root/autodl-tmp/SWE_project && PYTHONPATH=src \
  /root/autodl-tmp/conda/envs/sweforge/bin/python scripts/interop_mac.py --base-url http://127.0.0.1:8500
```

### 7.2 启动与监控

```bash
# 启动（后台）: 见 §3.3 命令; 终端输出全量 tee 到 $RUN_DIR/train.log
tail -f $RUN_DIR/train.log                       # 实时日志
nvidia-smi                                       # 期望 4 卡各 ~11GB（FSDP 6.8 + vLLM ~3）
df -h /root/autodl-tmp                           # 保存后检查 ≥20GB 空闲
# 产物全部落项目路径 $RUN_DIR/（2026-08-10 起约定）:
#   train.log（终端全量输出）/ metrics.jsonl（verl FileLogger）/
#   tensorboard/（TensorBoard 事件）/ grpo_training_curves.png（曲线图）/
#   rl_prompts_{train,val}.jsonl / checkpoints/（--save 时）;
#   verl hydra 配置另落 outputs/<date>/<time>/.hydra（torchrun 自动生成）
# 关键进度标记（grep）:
#   update_weights done / step:N - training/global_step:N /
#   timing_s/agent_loop/generate_sequences / compute_score / update_actor
```

### 7.3 正常退出判定

`training/global_step` 达到目标后，尾部出现 `DataLoader worker (pid ...) is killed by signal: Killed` 是 **verl/ray/torchrun 清理阶段的已知无害竞态**（cgroup oom_kill=0 非内存，step 指标已完整发出）。GPU 无残留、无 checkpoint 外残留即干净。

### 7.4 清理与故障排查

| 症状 | 处理 |
|---|---|
| **`Could not override 'max_model_len'`（hydra）** | EXTRA_ARGS 用了裸键。必须完整点路径：`actor_rollout_ref.rollout.max_model_len=4096`（§3.3），不是 `max_model_len=4096` |
| **vLLM `No available memory for the cache blocks`** | EXTRA_ARGS 缺 `actor_rollout_ref.rollout.tensor_model_parallel_size=4` → 单卡载全量 8GB 权重 + u=0.2 无余量给 KV cache（2026-08-11 实测） |
| GPU 有 Ray/vLLM 残留 | `/root/autodl-tmp/conda/envs/sweforge/bin/ray stop --force` |
| Mac 连接失败 | 先查 Mac server 端口，再查隧道；**不要切 Mock** |
| **reward 全 0（先查工作区再查模型）** | ① 空工作区：Mac bundle 查找是否带后缀剥离补丁（§5）？142 全覆盖探针（`/tmp/probe_full_142.py` 思路：带后缀 create → 数 /workspace py 文件）是否全命中？② 工具调用是否真实执行（`action_trace`/tool_call_count）③ verifier 结果——**先不调 lr** |
| **`CUDA error: invalid device ordinal`（init_model 阶段）** | **多 driver 症状**（NPROC_PER_NODE=4 时 4 个 torchrun rank 各起 ray 集群 → 设备 ordinal 错乱 + 线程 EAGAIN 整群 worker 死亡）。**必须 NPROC_PER_NODE=1 + EXTRA_ARGS `trainer.n_gpus_per_node=4`**（2026-08-10 v2 实测） |
| `unknown arg: --no-save` | 脚本只认 `--save N`；默认即不保存（NO_SAVE=1），去掉该参数即可 |
| OOM 在 update_weights/state_dict | 确认 LOAD_FORMAT=safetensors + LAYERED_SUMMON=true 生效（缺任一项 4B 必 OOM）；micro batch 不是解法 |
| 线程 EAGAIN（OpenBLAS） | 确认单 driver（NPROC=1）+ 脚本内线程限制生效 |
| 磁盘满 | 检查 `/`（系统盘仅 13G）；训练产物落项目路径 `outputs/<date>/`（同盘 `/root/autodl-tmp/SWE_project/outputs`） |

### 7.5 预检通过标准（阶段 ①）

四卡被 veRL 使用且只有一个训练 driver；加载 step_136 Adapter；日志出现 update_weights / generate_sequences / compute_score / update_actor / global_step:1；Mac 收到真实 Docker create/tool/verify 请求；进程干净退出。**reward 全同只说明样本无组内差异，不算链路失败。**

---

## 8. 资源预算（正式 50 step 预估）

| 项 | 值 | 依据 |
|---|---|---|
| GPU 显存 | ~11GB/卡（峰值 15.5GB/24GB） | 预检实测 FSDP 6.82GB + vLLM ~3GB |
| CPU 内存 | 预检峰值 127GB/240GB（容器限额） | FSDP CPU offload + DataLoader |
| 磁盘 | ckpt 每份 ~8–16GB（FSDP 分片 + optimizer）；step25/50 共 2 份 | 保存后 `df -h` 确认 ≥20GB 空闲 |
| 单 step 耗时 | toy 预检 52s（prompt 295）；正式 prompt 3072 预计 2–4 分钟 | rollout 每任务 4 条 × Mac Docker 串行/并行 |
| 总时长 | 50 step ≈ 2–3.5 小时 | 不含首次模型加载（~3 分钟） |
| 收敛判定 | resolve rate / reward 方差 / KL / 工具调用行为，50 step 后评估是否续 100 | 不预设盲目长训 |

---

## 9. 铁律清单（违反 = 架构失败）

1. SFT 与 RL 共用唯一 Canonical Trajectory/Tool Protocol，`validate_trajectory()` 校验。
2. 统一 mask：system/user/tool=0，assistant=1。
3. RL 保留真实 prompt_ids/response_ids/response_mask/logprobs，禁止事后重新 tokenize。
4. Agent Loop 禁止依赖 Docker（只走 EnvironmentBackend）。
5. 不伪造 verifier reward；policy-visible TaskSpec 禁止 gold/hidden/mutation。
6. 不升级 torch/veRL/vLLM/transformers/CUDA（锁版本 baseline）。
7. GRPO 禁止 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
8. torchrun 单 driver（NPROC=1 + n_gpus_per_node=4），不启动四份 PPO driver。
