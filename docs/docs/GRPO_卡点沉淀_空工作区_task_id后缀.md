# GRPO 卡点沉淀：reward 恒 0 的真根因 —— task_id uuid 后缀打爆 Mac bundle 查找

> 版本：2026-08-10。沉淀对象：4B + SFT LoRA（step_136）正式 Agentic GRPO 的 reward 恒 0——
> 环境、数据、生成器都验证"健康"后 reward 依旧恒 0。最终定位到 **AutoDL 侧
> `verl_agent_loop.py:150` 给 task_id 加 uuid 后缀，Mac server 按完整 task_id 精确匹配
> `bundles-dir/<task_id>`，永远 miss → 空模板容器 → 模型在空仓库里 search/view 全空 →
> 空 patch → reward 恒 0**。
>
> **结论翻转声明**：本文件是旧文档 `GRPO_卡点沉淀_SFT零工具调用.md` 的重写。旧文档归因
> "SFT 训练数据零工具调用格式"——**该结论错误**（详见 §3）。用户持 SFT eval 记录
> （工具调用可解析率 0%→100%→100%）质疑后复核推翻。
>
> 姊妹篇（按序排查，本卡点是第三层）：
> - `docs/GRPO_卡点沉淀_Docker环境.md` — 第一层：bundle 缺失/数据质量
> - `docs/SWE-Forge_formal任务Docker空工作区问题复盘.md` — 第一层复盘（bundle 未部署）
> - `docs/GRPO_卡点沉淀.md` — OOM / 多卡适配 / verl 边界
>
> 三篇文档的共性症状都是"reward 恒 0 + 空工作区"，但**三层根因不同**：bundle 没部署 →
> 数据语义坏 → **本卡点：bundle 在 Mac 上好好的，但查找键永远不对**。

---

## 一、现象：环境修好后 reward 依旧三恒 0

- 2026-08-10 预检（`grpo_4b_formal_v1`，Qwen3-4B + step_136 adapter，16 rollout × 2 step）：
  `rewards/mean=0.0`、`advantages/mean=0.0`、`pg_loss=0.0`——**完全没有学习信号**。
- 与第一层卡点（bundle 缺失）的**关键区别：环境看起来已经修好**——bundles_v2 已部署、
  gold-patch 探针 3/3 `resolved=1.0`、空 patch 3/3 `unresolved=0.0`。reward 公式、容器、
  测试全部"真实有效"。
- 训练本身"健康"且模型**真实在干活**（step 指标实录）：

  | 指标 | step1 | step2 | 说明 |
  |---|---|---|---|
  | num_turns | 4.31 (min3/max5) | 4.75 | 模型在多轮里真实试错 |
  | response_length/mean | 859 | 963 | 全程生成到接近上限 |
  | timing_s/agent_loop/tool_calls/mean | 1.01s | 1.17s | **工具真实执行**（到达 Mac Docker） |
  | timing_s/agent_loop/compute_score/mean | 0.57s | 0.53s | CleanVerifier 真实调用 |
  | pg_loss / rewards / advantages | 0.0 / 0.0 / 0.0 | 同左 | **无学习信号** |
  | actor/perf/max_memory_allocated_gb | 9.92 | 10.02 | 无 OOM（4B LoRA 正常） |

- 结论：链路每一环都"在工作"，模型也真在调用工具——但 reward 恒 0。"每环都工作"恰恰
  是最大的迷惑项：**唯一坏掉的一环（工作区是空的）被所有健康指标掩盖了**。

## 二、错误归因 vs 正确归因

| 阶段 | 归因 | 判定 |
|---|---|---|
| 前期 | "64-token 每轮截断，JSON 装不下" | **半对**：真 bug，修复后行为恢复（num_turns 2.0→4.31，工具真实执行），但 reward 依旧 0 → 还有下层根因 |
| 中期 | "SFT 训练数据零工具调用格式，模型无从学起" | **错误**（本文件旧版结论）。复核后：tool_calls 字段在 SFT assistant 消息里 **2771/2771** 存在；模型 100% 能输出可解析工具调用（§3） |
| 中期 | "没有格式奖励，模型学不会" | 方向相关但非根因：模型其实**会**格式 |
| 正确 | **task_id uuid 后缀 → Mac bundle 查找永远 miss → 空模板容器** | 证据链见 §4-5：决定性双探针对照 |

## 三、被推翻的假设（用户的证据）

### 3.1 "SFT 没有工具调用格式" —— 错

用户提供的 SFT eval 记录（`eval_sft_checkpoint.sh`，greedy decode）：

| step | 工具调用可解析率 | 首工具与示范一致率 |
|---|---|---|
| 早期 | 0% | 0% |
| 中期 | **100%** | 68.75% |
| step_136 | **100%** | 43.75% |

复核数据（`experiments/sft_qwen3_4b_lora_r16_e1_20260809/data/train.jsonl`，2904 条）：
- assistant 消息的 **messages 层 `tool_calls` 字段**（OpenAI 风格 `{"id": "call_0",
  "name": "search", "arguments": {...}}`）存在率 **2771 / 2771（100%）**；
- 旧结论只 grep 了 `content` 字符串（散文），**漏掉了 messages 层字段**——content 里
  确实没有 JSON，但工具调用本就不该在 content 里。

### 3.2 "模型不会 JSON 工具调用语法" —— 错

受控实验（`/tmp/verify_sampling.py`，4B+step_136，parse_tool_call_text 判定）：

| 组合 | 描述 | parse 成功率 |
|---|---|---|
| A | greedy + SFT holdout prompt | 8/8 |
| B | greedy + GRPO 真实 prompt | 4/4 |
| C | **temp=1.0/top_p=1.0 + GRPO 真实 prompt（GRPO 实况复现）** | **4/4** |
| D | temp=0.3 + GRPO 真实 prompt | 4/4 |

模型在 GRPO 的采样条件下 100% 输出可解析工具调用——采样、prompt、格式均非根因。

## 四、决定性探针（真实出问题的例子）

同一任务 `toolz-s16-remove_conditional-003`（toolz 仓库，`_signatures.py:741` 删除
`if func not in signatures: return None`；base 态真实存在字符串 `has_unknown_args`，
在 `_signatures.py` docstring + `functoolz.py` 等多处），两次 create 只差 task_id：

### 4.1 探针 A：task_id 带 `-probe5t` 后缀（生产同款路径）

```
task_id = toolz-s16-remove_conditional-003-probe5t
create → env: toolz-s16-remove_conditional-003-probe5t（成功, 无报错）
bash ls /workspace      → 只有 .git 和 .gitignore（空 git 模板!）
search "has_unknown_args" → matches=[]
view_file toolz/_signatures.py → file not found
export_patch             → 空
```

### 4.2 探针 B：task_id 原始（无后缀）

```
task_id = toolz-s16-remove_conditional-003
create → env: toolz-s16-remove_conditional-003（成功）
bash ls /workspace      → 完整仓库（.github/ toolz/ tests/ 等）
search "has_unknown_args" → 命中 5 处（_signatures.py:4、functoolz.py:228/317/324、tests）
view_file toolz/_signatures.py → 正常返回 783 行
```

**一个变量的差距，从"什么都有"到"什么都没有"——search/view_file/bash 工具本身完全
健康，坏的是容器里的内容。** 这也解释了此前所有困惑：gold-patch 探针 3/3 通过是因为
探针用原始 task_id（bundle 命中）；正式 GRPO 每次 rollout 都带后缀（bundle 永远 miss）。

## 五、为什么会有后缀（设计意图 vs 契约漏洞）

### 5.1 后缀的来源（`src/sweforge/rollout/verl_agent_loop.py:150`）

```python
# 唯一 env instance id: 同一 task_id 的 G 条 rollout 并发也不互相覆盖
task = task.model_copy(update={"task_id": f"{task.task_id}-{uuid4().hex[:12]}"})
```

意图正确：同一任务的 G 条 rollout 并发执行，env 注册表 key 必须唯一。

### 5.2 契约的漏洞（task_id 一身两职）

§8 契约（`src/sweforge/environment/server.py` create_env）：

```python
envs[task.task_id] = backend.create(task)
return {"env_id": task.task_id}        # env_id == task_id
```

Mac server 的 `backend.create(task)` 用 `task.task_id` 直接查找 bundle
（bundle 目录名 = task_id，见 `scripts/build_ast_bundles.py:114`
`bundle = bundle_root / str(task["task_id"])`）：

```
bundles-dir/<task_id>          ← 精确匹配, 无后缀剥离逻辑
```

**task_id 同时承担两个职责：env 注册表 key（需要唯一 → 后缀必要）+ bundle 查找 key
（需要原始 → 后缀致命）。** 两职合一且两侧没有同步约定时，后缀必然打爆查找。
设计时隐含假设"Mac 侧 bundle 查找不受后缀影响"或"server 会剥离后缀"——都不成立。

### 5.3 为什么 verify 不受影响（掩盖了 create 的问题）

`/v1/verifications` 的 task 是**内联 payload**（tests/image/setup_commands 全在
TaskSpec 里，`remote.py` verify 发送 `body={"task": task.model_dump(...)}`），不走
bundle。所以 CleanVerifier 用任务自带测试真实判定 unresolved——它确实"真实有效"，
只是永远在判一个空 patch。**验证链的健康掩盖了工作区链的坏死。**

## 六、修复

### 6.1 Mac server（vendored, 必须改）—— bundle 查找加后缀剥离回退

`backend.create()` 里 bundle 查找改为：先精确匹配，失败则剥离尾部
`-[0-9a-f]{12}`（12 位 hex = `uuid4().hex[:12]`）再查；两者皆无才走现有空模板回退：

```python
import re

_BUNDLE_SUFFIX = re.compile(r"-[0-9a-f]{12}$")

def _bundle_dir(task_id: str):
    for candidate in (task_id, _BUNDLE_SUFFIX.sub("", task_id)):
        d = bundles_dir / candidate
        if d.exists():
            return d
    return None  # 两者皆无 -> 沿用现有空模板回退
```

语义安全：剥离后查不到就不回退剥离结果（bundle 存在性双重确认），绝不影响原始
task_id 的查找行为；env 注册表唯一性不受影响（key 仍是完整带后缀的 task_id）。

### 6.2 AutoDL 侧——不改行为，钉死契约

- 后缀格式 `-{uuid4().hex[:12]}` 由新增回归测试
  `test_unique_env_suffix_format` 断言（`-[0-9a-f]{12}$`），防止两侧失配漂移；
- `verl_agent_loop.py:149` 加契约注释，指明 Mac 侧按此剥离。

### 6.3 修复后预期

生产 rollout 拿到真实代码快照 → search/view_file 有内容 → 模型至少能读代码、
尝试 str_replace → 不再"必然 0"。4B+step_136 真实修复成功率是模型能力问题
（reward 可能有方差），但链路层面的必然 0 解除。

## 七、历史遗留仍有效

- **64-token 截断修复**（`_turn_budget` 读 `multi_turn.max_tool_response_length`，
  缺失回落 256）仍然有效——预检 num_turns 2.0→4.31 就是修它之后出现的，模型才有机会
  多轮试错。
- **检查顺序更新**（三篇文档合起来）：环境（bundles 存在性/gold 探针）→ 数据质量
  （pool 语义/测试可执行）→ 生成器（64 截断/预算）→ **env 创建与 bundle 查找的一致性
  （本卡点）** → 最后才轮到模型能力。

## 八、方法论沉淀

1. **用户的指标优先于我的 grep**。旧结论"零格式"只查了 content 字符串，漏掉 messages
   层 `tool_calls` 字段；用户拿 eval 记录（100% 可解析率）一质疑就推翻。数据体检必须
   检查结构字段，不能只 grep 文本。
2. **"链路每环都工作"不等于"链路工作"**。工具真实执行、验证真实调用、容器真实创建——
   但工作区是空的。健康指标是迷雾，唯一的分辨手段是对照探针。
3. **对照探针是决定性的**。一个变量（task_id 后缀）两发 create，从"什么都有"到"什么都
   没有"，直接定性，不用猜。
4. **两职合一的 key 是静默炸弹**。task_id 兼 env 注册表 key 与 bundle 查找 key，职责
   分裂必须两侧同步约定（本卡点的教训：约定缺失 + 单侧改动 = 静默断链）。

---

## 附：本卡点关键文件

| 文件 | 角色 |
|---|---|
| `src/sweforge/rollout/verl_agent_loop.py:150` | 后缀来源（env 并发唯一）；149 行注释钉契约 |
| `src/sweforge/environment/server.py` create_env | §8 契约：`env_id == task.task_id`（两职合一） |
| `scripts/build_ast_bundles.py:114` | bundle 目录名 = task_id（Mac 查找的键） |
| `/tmp/probe_five_tools.py` + `/tmp/probe_nosuffix.py` | 决定性双探针（后缀 → 空工作区 / 原始 → 完整代码） |
| `/tmp/verify_sampling.py` | 4 组合受控实验（模型 100% 可解析, 推翻采样/格式假设） |
| `tests/test_verl_agent_loop.py::test_unique_env_suffix_format` | 后缀格式回归测试（新增） |
| `experiments/grpo_4b_formal_v1/train.log` | 预检 step 指标（三恒 0 + 工具真实执行） |
