# GRPO reward 恒 0 卡点沉淀：SFT 训练格式与 rollout 协议的差距（2026-08-11）

> 系列文档之一（前序: `GRPO_卡点沉淀_空工作区_task_id后缀.md`, `GRPO_卡点沉淀_Docker环境.md`）。
> 本文档修正了诊断过程中一个被推翻的结论（§2.3），并给出修正后的根因与修复方向。

## 1. 现象

阶段③ 50-step 正式训练（`outputs/2026-08-10/grpo_4b_formal_v2/`）：
Qwen3-4B-Base + SFT LoRA adapter（step_136, r16/α32）+ 142 条 Mixed Pool，
`--steps 50 --batch 4 --n 4`。运行 8 步后检查学习信号：

| 指标（8 步全部） | 值 |
|---|---|
| `critic/rewards/mean` | **0.0**（恒 0） |
| `actor/pg_loss` | **0.0**（恒 0） |
| `actor/grad_norm` | **0.0**（恒 0） |
| `actor/entropy` | 0.77 – 0.89（非 0，policy 分布有效） |
| `perf/time_per_step` | 432.7s → 339.1s（稳态 ~5.7min/step） |

entropy 非 0 而 pg_loss 恒 0 ⟹ advantage 恒 0 ⟹ **policy 零更新，训练空转**。
预估算完 50 步约 4.5h，但无任何学习信号，停机诊断。

## 2. 诊断链

### 2.1 第一层：单条 rollout 文本诊断（probe1: 轮预算 256 / 响应总预算 1024）

精确复刻 `verl_agent_loop.py` 回合逻辑（vLLM 0.11 + TokensPrompt + LoRA，
任务 r2e-aiohttp-74a2002b63，Mac Docker 真实执行），完整回合如下：

**turn 0**（生成打满 256 token）：
```
'I'll help you resolve this issue by first examining the relevant code in the
web application framework. Let's start by searching for files related to the
Application class and its finish event handling.

1. First, let's search for files containing "Application":
{"name": "search", "arguments": {"query": "class Application", "path": null}}
2. Let's look at the Application class implementation:
{"name": "view_file", "arguments": {"path": "web.py", "start_line": null, "end_l...
```
→ parse 取**第一个** JSON（search）→ 执行成功（observation 195 tokens,
`aiohttp/web.py:1259: ...`）。第二个 view_file JSON 在轮末被**截断**（`"end_l`）。

**turn 1**（生成打满 256 token）：
```
'...Let me continue searching for "finish_events":
{"name": "search", "arguments": {"query": "def finish_events", "path": null}}
{"name": "view_file", "arguments": {"path": "/web/application.py", ...}}
{"name": "str_replace", "arguments": {"path": "/tmp/tmpoakkuaft/web.py", ...
```
→ 一条消息塞 **3 个调用**，parse 只执行第一个（search）；第二个
`/web/application.py` 带**前导斜杠**被拒（path must be relative）——路径幻觉。

**turn 2**（str_replace JSON 截断）→ `PARSE_ERROR: ...no tool_call JSON found` 终止。

**终止态**：response 982/1024 tokens，**patch 0 chars**，verify `unresolved`
f2p=0 p2p=1。

### 2.2 第二层：预算放大实验（probe2: 轮预算 512 / 响应总预算 2048）

把每轮预算翻倍、总预算翻倍重跑同一任务，验证"预算不足"假说：

**turn 0**（打满 512）：同样塞 2 个调用+叙述，第二个 view_file JSON 截断
（`"end_l`），第一个 search 执行成功。
**turn 1**（打满 512）：塞 3 个调用，search 成功。
**turn 2**（打满 512）：**纯乱码**——无任何工具调用：
```
'𫟅
มนุ
ลัง
ฟัน
ครึ่ง
จับ
ไม่เคย
รักษา

僚
שׁ

ติดต่อ
เวิ
奈
หลักสูตร
...
<think>
กว้าง
บริษ
เกี่ยว
مَ
社
 לחלוט
บุรี
...
แฟชั่น

промышленн
เทคนิ
ทีม
ดูแล
มีโอกาส
ﲋ
ก็ได้
แบ่ง
ผลิตภั
ฤดูกา
...
'
```
（泰文 / 阿拉伯文 / 希伯来文 / 生僻字混排）→ PARSE_ERROR 终止。

**终止态**：response 1565/2048，patch 0 chars，verify unresolved。

**结论：预算不是根因。** 预算翻倍只是让模型在同一轮塞更多调用与叙述；
turn 2 的乱码与预算无关——是采样发散。

### 2.3 第三层：数据断层假设——提出与**推翻**（重要教训）

诊断 turn 0/1 有 JSON 形态、turn 2 崩坏、从不 finish，一度假设
"SFT 数据没教模型输出工具调用"。初期证据（**误导性**）：

- `train.jsonl` 的 `messages` 字段：assistant 消息 **0 个** `tool_calls`，
  纯叙述 + 独立 tool 角色消息 ⟹ 初步判定"SFT 从未训练 JSON 输出"。

**推翻证据（token 层，训练真正消费的层）**：

| 检查项 | 结果 |
|---|---|
| `train.jsonl` response_ids decode 含 JSON 调用 | **1880/1880 = 100%** |
| 含 `<tool_call>` 标记 | **2904/2904 = 100%**（每样本 7–42 个，均值 ~14） |
| response 含 `finish` 调用 | **2904/2904 = 100%** |
| 第一个 `<tool_call>` 就是 finish | 0/2904（finish 永远是**收尾**动作） |
| train.jsonl vs sft_dataset.jsonl 同 task token 序列 | prompt_ids/response_ids/message_token_counts **完全相同**（0 不一致） |

训练目标文本样例（decode `response_ids`）：
```
<|im_start|>assistant
I'll help you resolve this issue following the suggested steps. Let's start by
exploring the repository structure to locate relevant files.

1. First, let's explore the repository:
<tool_call>
{"name": "view_file", "arguments": {"path": "/testbed", "start_line": null, "end_line": null}}
```

**教训（写进本文档供后续诊断引用）**：SFT 加载路径铁律是"不重新 tokenize，
直接用落库 prompt_ids/response_ids"——**模型学到什么 = token 层说了算**，
`messages` 字段只是展示层重建（丢 `tool_calls` 结构、JSON 以文本形式存在
于 content 时正则未命中），不能作为"训练内容"的证据。判断训练内容必须
decode response_ids。

## 3. 根因（修正版）：训练格式与 rollout 协议的系统性差距

模型**确实学过** "叙述 + `<tool_call>`JSON + 多轮 + 收尾 finish" 的格式，
但 rollout 中不复现。五处差距 + 一个 harness 消费问题：

0. **harness 只消费模型文本中的第一个调用**（用户诊断方向，确认属实）：
   `parse_tool_call_text` 用 `_extract_json` 提取**恰好一个** JSON
   （parse.py docstring 明写"恰好一个结构化 tool_call"），模型一轮写出
   的 2–3 个调用**只有第一个被执行，其余全部丢弃**。而 SFT 训练目标
   **100% 每条 assistant 消息恰好 1 个调用**（4409/4409 统计）——模型
   的"一轮多调用"不是从训练学来的，是 rollout 自回归续写（训练中 JSON
   后紧跟 `<|im_end|>` 与 tool 消息，rollout 中 vLLM 无停止信号，模型
   写完第一个 JSON 后继续自回归写第二个）。**harness 丢弃 = 模型写出的
   动作信息损失**，模型"努力多走几步"的尝试全被浪费。

1. **标记层差距**：训练目标每个 JSON 都带 `<tool_call>` 标记（模板注入）；
   rollout 模型输出**裸 JSON**（无标记）。parse 以 `<tool_call>` 优先、
   `_extract_json` 兜底，裸 JSON 也能提取——但标记本身学而不牢，说明
   模型对格式边界的记忆脆弱。
2. **采样温度差距**：SFT 是 teacher-forcing 确定性目标；rollout
   `temperature=1.0` 采样。4B 在分布低概率区间直接发散为多语言乱码
   （probe2 turn 2 全段）。这是乱码的直接来源。
3. **轮预算截断**：模型叙述习惯长（"1. First, let's... 2. Let's look..."），
   256 token 内写不完"叙述 + 完整 JSON"，且倾向**一轮塞 2–3 个调用**
   （训练目标每轮 1 个）——JSON 频繁截断 → PARSE_ERROR 提前终止。
4. **finish 永不触发**：训练目标里 finish 100% 出现在轨迹**结尾**（0% 在
   首位）；rollout 中模型 2–3 轮就被截断终止，**从没走到该 finish 的收尾
   阶段**。截断终止 → patch 为空 → reward 0 → advantage 0 → 零更新。
5. **prompt 分布差距（domain gap）**：SFT prompt = R2E-Gym issue 格式；
   rollout prompt = TaskSpec 渲染问题描述 + sweforge system prompt（§18）。
   模型在未见分布上的泛化进一步拉低格式稳定性。

**因果链**：温度 1.0 + 预算截断 + 一轮多调用 ⟹ 从不出 finish / PARSE_ERROR
提前终止 ⟹ patch 空 ⟹ CleanVerifier 全 0 ⟹ advantage 0 ⟹ pg_loss 0 ⟹
policy 零更新。

## 4. 已落地修改

- `scripts/run_grpo_agentic.sh`：新增 `--turn-budget N` 参数，透传
  `actor_rollout_ref.rollout.multi_turn.max_tool_response_length`（verl
  默认 256；4B 建议 512）。总预算 `--max-response-length 2048`。
- probe2 验证了纯预算放大不解决根因（§2.2）。

## 5. 修复方向（进度更新 2026-08-11）

1. **harness 多调用消费（用户提议，已落地）**：`parse.py` 新增
   `parse_all_tool_calls`（`<tool_call>` 块 / ```json / 平衡 JSON 依次提取
   **全部**调用，max_calls 上限 8，截断/残缺候选跳过继续；全无效 →
   ParseError 保持"无合法调用即终止"），`verl_agent_loop.py` 主循环改为
   每轮顺序执行全部调用、observation 拼接回灌，finish 之后的调用不执行。
   局限（提取器边界）：文本层无法区分"嵌套对象"与"截断 JSON + 新对象"，
   截断在前时会吞掉后续（真实形态是截断在末尾，不影响）。
2. **采样侧（已落地）**：`actor_rollout_ref.rollout.temperature` 1.0 → 0.7
   （`run_grpo_agentic.sh --temperature` 可调）。
3. **数据侧（think block 方案，用户提议，暂缓）**：需重构 SFT 数据并重新
   训练，本次不动。方向保留：思考/叙述段 + `<tool_call>`JSON 调用段分离 +
   rollout 停止信号，从源头消除"一轮多调用"。
4. **预算（已落地）**：轮预算 512 / 总响应 2048（`--turn-budget`；
   验证过纯预算放大不解决根因）。
5. **验证**：单条 probe（多调用执行 + 温度 0.7）→ 确认后重启 GRPO 观察
   前几步 reward 非零率。

## 7. base vs SFT 生成对比（2026-08-11, 同 prompt 各 256 token）

**BASE（无 SFT）**——无任何工具调用形态，输出"代码审查报告"式文本+乱码尾：
```
After reviewing the code, I found that the issue is caused by a utility library
that clobbers the second argument on the callback. [ERROR DETAILS] ... 팹
ünstsystem усystem uni
```
（含 JSON: False | 含 `<tool_call>`: False | 含 finish: False）

**SFT step_136**——**JSON 调用形态从无到有**（SFT 的核心学习成果），
但无 `<tool_call>` 标记、无 finish、一轮塞多个调用、夹杂泰文乱码：
```
I'll help you resolve this issue with the application finish callbacks. Let's follow these steps:

1. First, let's search for files related to the application and finish callbacks
2. Then we'll examine the relevant code
3. Finally, we'll fix the issue if necessary
ประชาสัมพันธ์
เข้า
1. Let's search for files that might contain the finish callback registration code:
ครู
{"name": "search", "arguments": {"query": "register_on_finish", "path": null}}
ฟอร์ข้าว
 จึง
2. Let's view the web.py application file to find the registration code:
歷
{"name": "view_file", "arguments": {"path": "/web.py/application.py", ...}}
...
```
（含 JSON: True | 含 `<tool_call>`: False | 含 finish: False）

⟹ **SFT 学到了 JSON 调用的内容结构**（query 从任务提取、工具名/参数结构
正确），**没学到**：`<tool_call>` 包裹标记、finish 收尾习惯、单调用纪律、
无乱码稳定性。乱码是温度 1.0 采样放大的低概率行为（base 也有乱码尾）。

## 6. 与既有结论的关系

- 阶段 17 记录"0.6B 未学会 tool_call 语法（reward mean=0 是模型能力）"：
  与本次同根因家族——模型分布与 rollout 协议的差距；0.6B 是能力不足，
  4B+SFT 是泛化与采样问题，但机制相同（从不出 finish ⟹ reward 0）。
- `#86`（task_id uuid 后缀打爆 bundle 查找）是**另一条独立问题链**，
  已修复落地（`#89`）。本次文档是 reward 恒 0 的第二层（行为层）根因。

## 7. 行为层的训练侧解法（2026-08-11 v4, 已落地）

链路修好后面临的才是真正的训练问题: **纯二进制在"全部 unresolved"的 step
里组内 reward 全同 -> advantage 恒 0 -> pg_loss 恒 0 空转**（v3 后 step 1
实测 16/16 reward 0）。奖励侧解法 = F2P partial 混合奖励:

- `reward/verifier.py::partial_reward`: resolved → 1.0 锚不变; 未解决时按
  **逐条 F2P 测试真实通过情况**给部分分（`f2p_credit`）+ P2P 破坏惩罚
  （防"改烂文件让 F2P 全过"刷分）+ format 小台阶（工具调用纪律 bootstrap）。
  每分来自 CleanVerifier 真实 exit code, 不伪造（P5）。
- 效果机制: 同任务 4 条 rollout 修好 0/1/2 个测试 → 组内 reward 有方差 →
  advantage 非零 → 梯度流动（不依赖模型"学会 finish"这个前置）。
- 难度加权（w_i 按 test_id）与正式 curriculum 留作 v2 扩展: 需要每测试
  通过率的离线 profiling（现有 profiled.jsonl 只有任务级）; 任务级难度
  混合已由 priority_pool 提供（easy/unsolved 恒 0 不占采样）。
- 训练侧评估: 重启后看 `critic/rewards/mean` 是否>0 且逐步上升、分布是否
  从"全 0"变为"0.05~0.9 摊开"（reward_breakdown 的 f2p_passed 明细）。
