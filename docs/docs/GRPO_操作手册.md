# GRPO 操作手册（v24 参数定案版）

> 适用: 4×4090（24GB/卡，TP=4）上运行 SWE-Forge Agentic GRPO（Qwen3-4B + SFT LoRA adapter + Mac Docker CleanVerifier）
> 参数出处: v24 run5 实测定案（2026-08-11）；卡点与修改背景见 `docs/GRPO_v24_核心修改与卡点沉淀.md`
> 预计耗时: ~8-10 分钟/step（16 rollout × Mac Docker verify），50 step ≈ 7-8 小时

## 1. 前置条件

| 项 | 要求 |
|---|---|
| 硬件 | 4×4090 24GB（TP=4 必须 4 卡；单卡 24G 放不下 4B 训练） |
| 基座模型 | Qwen3-4B-Base 本地路径（`HF_HOME=/root/autodl-tmp/huggingface` 缓存 snapshot） |
| SFT adapter | LoRA ckpt 导出产物（如 `eval/adapters/step_136`，r16/alpha32），用 `export_lora.py` 从 FSDP ckpt 导出 |
| 任务池 | `pool_easy_v1`（30 AST + 50 Reversal，优先池） |
| Mac server | `curl http://127.0.0.1:8500/health` 必须可达（反向 SSH tunnel 到 Mac 8501）；不可达训练脚本 exit 2 拒绝启动 |
| Python | `/root/autodl-tmp/conda/envs/sweforge/bin/python`（绝对路径；非交互 bash 的 `conda activate` 报 CondaError） |

## 2. 参数表（v24 run5 定案）

### 2.1 环境变量（入口处）

```bash
unset PYTORCH_CUDA_ALLOC_CONF    # 铁律: CuMem 池禁 expandable_segments
NPROC_PER_NODE=1                 # 单 driver（多 rank 各起 ray → 线程 EAGAIN 整群死）
N_GPUS_PER_NODE=4                # 训练吃 4 卡
GPU_MEM_UTIL=0.20                # vLLM 显存份额（TP=4 权重分布后 0.2 足够）
LOAD_FORMAT=safetensors          # vLLM 自载基座权重 → base_sync_done=True，update 跳全量反分片
LAYERED_SUMMON=true              # 逐层 gather LoRA 参数（峰值=单层 ~0.2GB，非完整基座 8GB）
```

### 2.2 训练参数（-- 后的 EXTRA_ARGS 最后覆盖生效）

| 参数 | 值 | 为什么 |
|---|---|---|
| `trainer.n_gpus_per_node` | 4 | 多卡训练（单 driver） |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | 4 | vLLM TP=4 吃满 4 卡 |
| `actor_rollout_ref.actor.optim.lr` | 5e-7 | 4B+LoRA 保守 lr（GRPO 对 lr 敏感） |
| `actor_rollout_ref.actor.ppo_epochs` | 1 | 少 epoch 防过拟合离线 rollout |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 1 | **显存峰值控制核心**：每次梯度累积只 1 条序列，激活减半（step1 实测峰值 19.2GB） |
| `actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking` | true | 熵计算 chunking 省显存 |
| `actor_rollout_ref.actor.fsdp_config.entropy_checkpointing` | true | 熵 checkpointing 省显存 |

### 2.3 数据/rollout 参数（CLI flags）

| 参数 | 值 | 为什么 |
|---|---|---|
| `--steps` | 50 | 正式训练步数 |
| `--batch` / `--n` | 4 / 4 | 16 rollout/step（组内 4 条同任务，GRPO 需组内方差） |
| `--temperature` | 0.8 | **用户指定不改**（0.7 压制多样性，1.0 发散多语言乱码） |
| `--max-prompt-length` | 3072 | prompt cap |
| `--max-response-length` | 6144 | **响应预算 = v23 同款**；v24 单调用协议实测 mean 4086，2048 会 3 轮截死探索（run4 教训） |
| `--turn-budget` | 512 | 单轮生成预算（256 内写不完完整工具调用） |
| `--max-num-seqs` | 4 | rollout 并发 = 训练并发，避免 vLLM 排队 |
| `--max-batched-tokens` | 4096 | rollout 侧保守 |
| `--save` | 25 | step 25/50 各存一份 ckpt（默认不保存） |

### 2.4 自动推导（脚本内，勿手改）

| 参数 | 推导 | 值 |
|---|---|---|
| `max_model_len` | prompt + response | 9216 |
| `ppo_max_token_len_per_gpu` | prompt + response（铁律：只设 response 触发 verl AssertionError） | 9216 |
| `max_assistant_turns` | `configs/agent_loops.yaml` | **12**（24 轮 obs 累积 ~24000 序列打满 24.5GB cumem OOM；12 轮 obs ~8400 与 v23 画像一致） |
| 每轮 stop_token | `<\|im_end\|>` | 模型在 im_end 处停，每轮回到"消息结束、观察未至" |

### 2.5 数据参数（内部）

- pool → `rl_prompts_train.jsonl`（prepare_grpo 剥离 gold/mutation 的 policy_view；**全量任务进 jsonl**，DataLoader 每 step 采 batch 行轮换——不要 `--max-tasks BATCH`，那会把 42 任务池截到 2 个任务）
- val jsonl：`--seed 11`，`batch/2+1` 条

## 3. 启动

```bash
cd /root/autodl-tmp/SWE_project
bash scripts/run_grpo_v24.sh          # 终端里跑会自动 nohup 自守护
# 或直接:
bash scripts/run_grpo_agentic.sh \
  --model <模型本地路径> --adapter <adapter 路径> --pool <pool.jsonl> \
  --steps 50 --batch 4 --n 4 --temperature 0.8 \
  --max-prompt-length 3072 --max-response-length 6144 --turn-budget 512 \
  --max-num-seqs 4 --max-batched-tokens 4096 --save 25 -- \
  trainer.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=true \
  actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=true
```

启动后核对（`train.log` 的 torchrun 行）: `temperature=0.8` / `max_model_len=9216` / `ppo_max_token_len_per_gpu=9216` / `ppo_mini_batch_size=1`（EXTRA_ARGS 覆盖生效）/ `max_response_length=6144`。

产物（`--out-dir` 可覆盖，默认 `outputs/<date>/grpo-agentic-<time>/`）:
- `train.log`（终端全量 tee）+ 自守护日志 `outputs/logs/grpo-agentic-*.log`
- `metrics.jsonl` / `tensorboard/` / `grpo_training_curves.png` / `system.jsonl`
- `checkpoints/`（--save N 时）

## 4. 验证点（训练前 2 步内必须确认）

### 4.1 step 1→2 过渡（显存验证点，run2/3 崩溃处）

```bash
# 等 global_step:1 出现后取峰值
/root/autodl-tmp/conda/envs/sweforge/bin/python -c "
import json, time
rows = [json.loads(ln) for ln in open('<out-dir>/system.jsonl')]
peak = max(rows, key=lambda r: r.get('gpu_mem_used_mib', 0))
print(time.strftime('%H:%M:%S', time.localtime(peak['t'])), peak['gpu_mem_used_mib'], 'MiB')
"
```

判定（24GB 卡，上限 24564 MiB）:
- **安全**: < 23.5GB（v23 实证线）；run5 实测 **19.2GB**
- **危险**: > 24.5GB（run2/3 崩溃线）
- step 2 rollout 开始且无 `cumem` / `out of memory` 报错 = wake_up 通过

### 4.2 rollout 行解读（train.log）

```
[rollout] task=reversal-more_itertools- term=finish turns=7 clean=0/7 tools=6
          parse_err=0 dropped=15 resp=1458 score=1.0000 binary=1.0000
          patch=486 f2p=1/1 [search2/view_file3/str_replace1] editfail=-
```

| 字段 | 含义 | 健康信号 |
|---|---|---|
| `term` | finish / parse_error / response_cap / env_error / max_turns | 多数 finish |
| `turns` | 轮数 | 不长期 1-3（被 cap 截死） |
| `clean` | dropped_calls==0 且真实生成 im_end 的轮数 | 随训练上升 |
| `dropped` | 被丢弃的多调用数（v24 协议只执行第一个） | 可观测即可，不阻塞 |
| `parse_err` | 解析错误次数 | 随训练下降 |
| `resp` | 实际响应 token 数 | **不长期 = 6144 满预算**（撞 cap） |
| `score` / `binary` | 塑形 reward / resolved 二进制 | resolved 出现 = 学习信号 |
| `patch` / `f2p` | patch 字节数 / F2P 通过 | 有增长 = 有效学习 |
| `editfail` | `-` = **无编辑失败**（不是有失败！） | 少量即可 |
| `mix` | 工具组合 | 出现 search/view 先行探索 = 非盲猜 |

## 5. 故障排查

| 现象 | 根因 | 处理 |
|---|---|---|
| `cumem_allocator.cpp:62` OOM（update 后 wake_up） | 序列过长/激活过高打满 24.5GB（run2/3 实证：24 turns obs 累积 ~24000） | 降 `max_assistant_turns`（12）或 `ppo_mini_batch_size`（1）；查 system.jsonl 峰值确认 |
| `resp=2048 满预算`，3 轮截死 | response 预算低于模型实际消耗 | 恢复 6144（v23 同款）；**预算不因显存妥协**（显存靠 mini_batch/turns 控制） |
| 训练侧 `AssertionError`（ppo_max_token_len） | `ppo_max_token_len_per_gpu` 只设了 response 长度 | 必须 = prompt+response |
| 参数丢 adv_estimator 变 GAE/critic 崩溃 | EXTRA_ARGS 行内有空行/注释，`\` 续行链断 | 参数行内禁止空行/注释 |
| 整群 worker 线程 EAGAIN 死 | 多 driver（NPROC>1 各起 ray）+ OpenBLAS 默认 64 线程 | NPROC_PER_NODE=1 + 线程数 8 |
| `env_error: remote env unreachable ... timed out` | Mac server 短暂超时（隧道抖动） | 确认 `/health` 恢复；训练是否死亡看 cumem/Traceback，而非单条 env_error |
| rollout 全 `dropped=15+` 无动作执行 | 模型多调用文本（协议只取第一个） | 正常协议行为；关注 patch/f2p 是否出现 |
| 训练立即 exit 2 | Mac server 不可达 | 重建 SSH tunnel；不允许 Mock 冒充 |

## 6. 监控

```bash
# 实时
tail -f <out-dir>/train.log | grep -E "\[rollout\]|global_step|Traceback"
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 5

# 峰值（见 §4.1 脚本）
# 进度
grep -o "global_step:[0-9]*" <out-dir>/train.log | tail -1
grep -c "\[rollout\]" <out-dir>/train.log        # 16/step
grep -c "score=1.0000" <out-dir>/train.log      # resolved 数
```

## 7. 铁律（违反 = 架构失败）

1. 不升级 torch/veRL/vLLM/transformers/CUDA（锁版本 baseline；唯一例外 tensordict 0.10.0 + pyvers）
2. 不伪造 verifier reward；policy-visible TaskSpec 禁止 gold/hidden/mutation
3. Agent Loop 只走 EnvironmentBackend，禁止 Docker 依赖
4. 禁止事后重新 tokenize（im_end 环境补位 mask=0/logprob=0）
5. `unset PYTORCH_CUDA_ALLOC_CONF`
6. torchrun 单 driver
7. `ppo_max_token_len_per_gpu` = prompt+response
8. 参数行内禁止空行/注释
9. Mac server 不可达 exit 2，不冒充
10. v23 checkpoint 不使用不续训
