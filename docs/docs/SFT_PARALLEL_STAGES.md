# SFT 实验与并行阶段操作说明

## 1. 本次 SFT 配置

- 模型：`Qwen/Qwen3-4B-Base`，BF16 LoRA，`r=16`，`alpha=32`。
- 数据：原始 3,061 条多轮 Agent 轨迹；按 `task_id` 分为 train 2,904 / holdout 157，无任务重叠。train 经 `max_length=8192` 过滤后实际使用 2,176 条。
- loss mask：仅 assistant 回复和工具调用 token 参与交叉熵；system/user/tool observation 只做上下文。
- 四卡 batch：`train_batch_size=16`，`micro_batch_size_per_gpu=1`，4 GPU。每轮前反向处理 4 条轨迹，累积 4 轮后更新一次；`2176 / 16 = 136` 个 optimizer step，即 1 epoch。
- 优化：LR `2e-4`，3% warmup，AdamW，梯度裁剪 1.0，梯度检查点开启。
- 保存：仅 step 68 和 136，`max_ckpt_to_keep=2`；训练后导出 PEFT Adapter 并删除大分片。
- 日志：实验目录 `formal_e1_lr2e4/`，完整终端日志同步写入 `runs/sft/`。

训练内指标：assistant-only loss、grad norm、LR、每步有效 token、GPU 显存/利用率、磁盘和 checkpoint 占用。固定留出集比较 Base/step68/step136 的 NLL、PPL、token accuracy、tool-call parse rate 和 first-tool match rate；最后补 Mac Docker 真实任务解决率。

## 2. 完整流程与当前状态

| # | 阶段 | 状态 | 可否与 SFT 并行 |
|---|---|---|---|
| 1 | Canonical 消息/工具/TaskSpec/Trajectory 协议 | 已完成 | 不需要 |
| 2 | Mac Docker Environment Server + AutoDL 反向隧道 | 已复验 | 是 |
| 3 | AST mutation / repair reversal / public task 数据工厂 | 代码完成；正式规模化未完成 | 是，但需新的 seed repo |
| 4 | CleanVerifier、F2P/P2P、determinism、leakage 漏斗 | 框架和小规模验收已完成 | 是 |
| 5 | SFT 轨迹 normalize/tokenize 与 train/holdout 固化 | 已完成 | 不需要 |
| 6 | Base 留出集评测 | 已完成 | 已先行完成 |
| 7 | 4×4090 正式 LoRA SFT | 进行中，1 epoch/136 steps | 主任务 |
| 8 | step68/136 FSDP 合并、Adapter 导出/验证 | 脚本已就绪，等 checkpoint | 否 |
| 9 | Base/SFT 离线指标与 Docker 任务评测 | Base 完成，step68/136 待测 | 否 |
| 10 | RL pool 转换、repo-isolated split、secret scan | 小规模联调集已完成 | 是，已执行 |
| 11 | Difficulty profiling + mixed priority sampling | 代码完成；真实 4B policy 未跑 | 部分；policy profiling 等 SFT |
| 12 | 0.6B AgentLoop→Mac Docker→GRPO 最小闭环 | 已完成 | 不需要 |
| 13 | 4B + SFT Adapter 正式 GRPO | 未开始 | 否，依赖 SFT Adapter |
| 14 | RL 后任务解决率、reward/效率、消融 | 未开始 | 否 |
| 15 | 失败分类→数据飞轮调权→下一轮任务 | 代码/dry-run 完成，真实 policy 轮未开始 | 否，需要 RL rollout |

注：当前 RL 集只用于联调，共12任务（train 10 / eval 2），不能写成“正式规模化数据”。正式 GRPO 前仍需扩展到 3–5 个 seed repo、100–250 个 CleanVerifier-verified 任务。

## 3. SFT 期间已并行完成的操作

### RL 小规模数据

```bash
cd /root/autodl-tmp/SWE_project
bash scripts/prepare_rl_data_small.sh
```

产物：`/root/autodl-tmp/experiments/sft_qwen3_4b_lora_r16_e1_20260809/parallel/rl_data_v1/`下的 `pool_train.jsonl`、`pool_eval.jsonl`、`rl_train.jsonl`、`rl_eval.jsonl`、`split_manifest.json`、`overlap_report.json`。train 为 `demo_calc + aiohttp`，eval 为 `datalad`；secret scan 和四维重叠检查均通过。

### 数据/协议定向测试

```bash
PYTHONPATH=src /root/autodl-tmp/conda/envs/sweforge/bin/python -m pytest \
  tests/test_public_source.py tests/test_prepare_grpo.py tests/test_contamination.py -q
```

本次结果：`25 passed`。

### Mac Docker 生产链路复验

Mac 侧保持 Environment Server 监听 8500，建立反向隧道后，AutoDL 运行：

```bash
cd /root/autodl-tmp/SWE_project
PYTHONPATH=src /root/autodl-tmp/conda/envs/sweforge/bin/python \
  scripts/interop_mac.py --base-url http://127.0.0.1:8500 --task-id sft-parallel-probe
```

本次结果：health、五工具、patch export、CleanVerifier（F2P=1/P2P=1）和 3-turn trajectory 全部通过。

## 4. SFT 结束后的顺序

1. 确认 step68/136 checkpoint 完整，运行 `scripts/export_eval_sft_checkpoints.sh`导出 Adapter 并清理大分片。
2. 在同一固定留出集上并行评测 Base/step68/step136，选最佳 Adapter；无证据不盲目跑第 2 epoch。
3. 扩展并 CleanVerifier 验证正式 RL task pool，再做 difficulty profiling/priority sampling。
4. 用最佳 SFT Adapter 跑 4B GRPO 小步混合奖励验收，通过后才扩大 steps/task/rollout 数。
5. 做 RL 后 Docker 任务解决率、奖励消融和数据飞轮。
