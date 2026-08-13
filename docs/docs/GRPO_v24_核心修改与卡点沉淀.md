# GRPO v24：核心修改与卡点沉淀

> 版本: v24 正式训练（2026-08-11，run5 定案）
> 状态: 50 step 正式训练中，step 1-2 过渡验证通过（无 cumem OOM），已出现第一条 resolved（score=1.0）
> 配套: 操作手册见 `docs/GRPO_操作手册.md`

## 0. 一句话总结

v23 暴露出"模型冲满 6144 response 预算也到不了 finish"（满预算打满）、"多调用 token 污染 policy loss"、"parse_error 输出被丢弃"三个阻塞；v24 以**单调用 token 协议**（解析先行、截断写入、im_end 收尾判断）修复全部三处，并把显存峰值从 23.5GB 刀尖（run2/3 崩溃线 24.5GB）压到 19.2GB（step 1 实测），让大预算（6144）与训练稳定性同时成立。

---

## 1. 三处核心代码修复

### 修复 1: im_end 判定基于**截断后**的保留序列

**位置**: `src/sweforge/rollout/verl_agent_loop.py` — `_extend_through_turn_end()`（118 行）+ 主循环 349-401 行

**问题**: v23 用原始 `output.stop_reason == stop` 判定回合结束。但单调用协议按字符边界截断 token 前缀后，**模型生成的 im_end 可能被截断切掉**——保留序列末尾没有 im_end，而 vLLM 的 stop_reason 仍是 stop，回合被判"干净结束"。

**修复**:
- 截断后调用 `_extend_through_turn_end()`：只吸收空白 token 与**紧邻的** im_end（且仅当 logprobs 覆盖，否则环境补，**不伪造 logprob**）
- 回合结束边界 = 保留序列末尾是否为 im_end（`generated_ids[-1] == im_end_id`），**v20 的 stop_reason 豁免作废**——stop_reason=stop 但保留前缀无 im_end 时，环境补 `<|im_end|>`（mask=0, logprob=0，不作为模型动作训练）
- `turns_clean` 只统计 `dropped_calls==0 且模型在保留序列中真实生成 im_end` 的回合

**效果**: "模型是否干净收尾"成为真实测量；环境补的 im_end 不参与 policy loss，token 纪律（mask: system/user/tool=0, assistant=1）不破。

### 修复 2: parse_error 不丢弃模型输出，全量写入

**位置**: `verl_agent_loop.py` 358-373 行

**问题**: v23 的 ParseError 路径只记录计数就 break——错误输出被扔出 policy loss，模型对"产生解析错误的 token"得不到负梯度，错误模式无法被惩罚纠正。

**修复**: ParseError 时**全量**写入 response_ids（mask=1 + 对应 logprobs），负奖励作用于真正产生错误的 token，然后 break 进终止流程。

**效果**: 解析错误 token 参与 GRPO，负 advantage 直接作用于错误文本本身。

### 修复 3: 候选提取按文本**实际起始位置**取最早

**位置**: `src/sweforge/rollout/parse.py` — `_extract_next_json()`（203-226 行）+ `_extract_balanced_object_pos()`（235 行起，新增位置版签名）

**问题**: 旧实现按固定优先级（`<tool_call>` 块 > ```json 代码块 > 平衡对象）返回第一个命中的候选——若文本更早位置是平衡 JSON 对象、`<tool_call>` 块在后面，仍会选后者，**丢早取晚，模型动作顺序被错排**（截断时后面那个甚至可能是残缺候选）。

**修复**: 三路各自定位，按**起始位置**取最早；起始位置相同时按 块 > 代码块 > 平衡对象（确定性 tie-break）。

---

## 2. 显存峰值卡点（v24 特有，实证链完整）

### 2.1 卡点现象

v24 协议让模型能真正冲满轮数上限后，**run2/3 连续 cumem OOM**：

```
cumem_allocator.cpp:62  (vLLM CuMemAllocator wake_up create_and_map 分配失败)
```

SystemSampler 时间线实证（`system.jsonl`，2s 采样）:

| 时刻 | 显存 | 事件 |
|---|---|---|
| 18:32:48-52 | 23772 → **24562 MiB** | run2 update 激活爬升打满 24.5GB（上限 24564） |
| 18:52:15 | — | run3 同点崩溃 |

对比: v23 同配置 6 步峰值 23507 MiB（余量仅 1GB，**侥幸**不崩）；run2/3 全满必崩。

### 2.2 根因链（完整因果）

```
max_assistant_turns=24（v6 时代为 SFT 轨迹 15.8 轮设的上限）
  → v24 截断协议下模型每轮 ~200-400 token 冲满 24 轮
  → observation token（mask=0）累积: update 序列 ~24000
    （prompt 3072 + resp 4812 + obs 23×~700）
  → FSDP 激活 16.7GB 动态大头（与序列长度×batch 成正比）
  → vLLM sleep（释放显存给 update）后 wake_up 需重新 create_and_map
  → 无空间 → cumem OOM（run2 崩在 18:32:52）
```

关键事实（verl 0.8.0 源码只读确认）:
- FSDP 路径（`_generated_ppo_trainer.yaml`）**没有 activation_checkpointing 配置**——该配置只在 AutomodelEngineConfig（新引擎 549 行）；显存峰值控制杠杆 = 序列长度 + ppo_mini_batch_size
- 显存组成: FSDP 权重 6.8GB 静态 + 激活 16.7GB 动态大头；LoRA optimizer state 极小（~0.3GB），optimizer_offload 无意义

### 2.3 解法（run5 定案，step 1 实测验证）

| 杠杆 | 值 | 作用 |
|---|---|---|
| `max_assistant_turns` | 24 → **12**（`configs/agent_loops.yaml`） | obs 累积从 ~24000 压到 ~8400，与 v23 显存画像一致 |
| `ppo_mini_batch_size` | 4 → **1** | 每次梯度累积只 1 条 rollout 序列，激活峰值减半 |
| response 预算 | **6144**（不砍） | 预算不因显存妥协（见 §3） |
| 铁律 7 | `unset PYTORCH_CUDA_ALLOC_CONF` | CuMem 池禁止 expandable_segments（手册 §3.1） |

**实测结果（step 1，run5）**:

| 阶段 | 显存 |
|---|---|
| rollout 期（14 分钟） | 稳定 **10.3 GB**（vLLM sleep） |
| update 期（~30s） | 16.4 → **19.2 GB** 爬升 |
| update 完成 | 回落 **10.4 GB** |

- step 1 全程峰值 **19160 MiB**（上限 24564，余量 5.4GB / 22%）
- 比 v23 的 23507 低 **4.3GB（-18%）**——mini_batch 1 激活减半生效
- step 2 wake_up（run2/3 崩溃点）通过，cumem OOM = 0
- 余量空间可用于后续上调 batch（4→6 预估峰值 ~22GB 仍安全）或流程预算

---

## 3. 预算卡点（run4 教训，用户红线）

**现象**: run4 把 response 预算砍到 2048（照抄 `docs/GRPO_正式运行手册.md` 的旧配方）后，模型 3 轮内撞 cap 强制终止——rollout 显示 `resp=2048`，模型 search2/view1 后到不了 finish/patch 决策点，探索空间被截死。

**教训**: **响应预算是上限不是消费目标，但也不能低于模型实际消耗**。v24 单调用截断协议下模型实际消费 mean ~4086 token（run1 实测），2048 会让流程在 3 轮内被 RESPONSE_CAP 截死。手册是给手册时代模型行为（resp mean 2005）写的，**参考 ≠ 照抄**——模型行为变了，预算必须跟着变。

**定案**: response 6144（v23 同款），max_model_len 自动 = prompt+response = 9216，显存安全靠 mini_batch 1 + turns 12 而非砍预算。

---

## 4. run5 配置快照（50 step 正式训练，2026-08-11）

```bash
NPROC_PER_NODE=1 N_GPUS_PER_NODE=4 GPU_MEM_UTIL=0.20 LOAD_FORMAT=safetensors LAYERED_SUMMON=true \
  bash scripts/run_grpo_agentic.sh \
  --model  <Qwen3-4B-Base 本地 snapshot> \
  --adapter <sft step_136 LoRA> \
  --pool   <pool_easy_v1/pool.jsonl> \
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

完整入口: `scripts/run_grpo_v24.sh`（含全部注释与决策理由）。

**学习信号实证**: step 1-2 出现第一条完全成功的 rollout——
`reversal-more_itertools-*`: search2 → view_file3 → str_replace1（无 editfail）→ finish，patch=486，F2P 1/1，score=1.0。模型在 16 条组内对比中，1.0 路径被 GRPO advantage 放大（v24 验收基线: 1/8 patch、F2P 0/8）。

---

## 5. 铁律（本版本不得违反）

1. 不升级 torch/veRL/vLLM/transformers/CUDA（锁版本 baseline；唯一例外 tensordict 0.10.0 + pyvers）
2. 不伪造 verifier reward；policy-visible TaskSpec 禁止 gold/hidden/mutation
3. Agent Loop 只走 EnvironmentBackend，禁止 Docker 依赖
4. 禁止事后重新 tokenize：保留真实 prompt_ids/response_ids/response_mask/logprobs；im_end 环境补位必须 mask=0/logprob=0
5. `unset PYTORCH_CUDA_ALLOC_CONF`（CuMem 池禁 expandable_segments，手册铁律 7）
6. torchrun 单 driver（NPROC_PER_NODE=1 + trainer.n_gpus_per_node=4）
7. `ppo_max_token_len_per_gpu` 必须 = prompt+response（只设 response 触发 verl AssertionError）
8. 参数行内禁止空行/注释（`\` 续行链会断，丢 adv_estimator → 默认 GAE → critic 路径崩溃）
9. Mac server 不可达即 exit 2，不冒充
10. v23 checkpoint 不使用不续训（诊断实验产物）
