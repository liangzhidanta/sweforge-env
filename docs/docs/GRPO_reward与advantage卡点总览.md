# GRPO reward=0 / advantage=0 卡点总览（按"系统侧"分类，2026-08-11 v2）

> 本文件把全部卡点按**系统侧**归类（v2 重构，替代 v1 的"两条主线"结构）。
> 每个卡点的症状都是 reward=0 / advantage=0，但**哪个侧坏了决定修哪里**。
> 配套: 操作手册 `docs/GRPO_操作手册.md` | v24 修改 `docs/GRPO_v24_核心修改与卡点沉淀.md`

---

## 0. 系统六侧与卡点归属总图

```
模型(4B+LoRA)
   │  ← C. 模型输出协议侧（parse/单调用/停止信号/温度预算）
   ▼
AgentLoop (AutoDL)
   │  ← B. Harness/契约侧（§8 契约/env 生命周期/bundle 查找键/观测链路）
   ▼
Mac Environment Server
   │  ← A. Docker 侧（容器内容/bundle 资产/执行器/测试运行）
   ▼
CleanVerifier ──→ reward ──→ D. 训练/算法侧（GRPO advantage/显存）
                              ▲
                          E. 数据侧（SFT/pool 输入原料）
                          F. 模型能力侧（能力边界, 非 bug）

卡点分布:  A×2  B×4  C×3  D×6  E×2  F×2（含跨侧卡点）
```

---

## A. Docker 侧（Mac 环境执行层）——2 个卡点

> 特征：容器内容不对、文件写不进去、测试没跑。改的是 **Mac 侧代码/资产**。

### A1. bundle 缺失 → Docker 空工作区
- **现象**: 容器里只有 `.git` 空模板, search/view 全空 → 空 patch → reward 0
- **根因**: bundles 未部署到 Mac server 的 `--bundles-dir`
- **解决**: bundles 部署 + gold-patch 探针 3/3 resolved 验证
- **详细**: `GRPO_卡点沉淀_Docker环境.md` + `SWE-Forge_formal任务Docker空工作区问题复盘.md`

### A2. str_replace 目录缺失 → ASGI 500 → 整条 rollout 死亡
- **现象**: 模型**正确意图**的编辑（path 所在目录不存在）→ `write_text` OSError →
  ASGI 500 → execute 异常 → ENV_ERROR → reward 0（harness 问题, 不是模型问题）
- **根因**: canonical 只说"文件不存在则创建", 没说**目录**不存在怎么办——执行器无兜底
- **解决**: Mac `mkdir -p` 兜底 + 文件系统类错误转 observation（跨到 B 侧的错误语义规则:
  只有基础设施故障才 500）
- **详细**: `GRPO_卡点沉淀_Mac_str_replace_mkdir.md`

---

## B. Harness/契约侧（AutoDL ↔ Mac 交互层）——4 个卡点 ★卡点最多的一侧

> 特征：键对不上、env 互杀、错误被误判、模型**看不到**错误。改的是契约双方。
> 这一侧是"链路每环都健康但 reward 恒 0"的重灾区——健康指标全是迷雾, 唯一分辨手段是对照探针。

### B1. task_id uuid 后缀打爆 bundle 查找 ★决定性卡点
- **现象**: bundle 部署好、gold 探针通过后 reward **依旧**三恒 0
- **根因**: `verl_agent_loop.py:150` 给 task_id 加 `-uuid12` 后缀（env 并发唯一）,
  Mac server 按完整 task_id 精确匹配 `bundles-dir/<task_id>` 永远 miss → 空模板容器。
  **task_id 一身两职**: env 注册表 key（要唯一）+ bundle 查找 key（要原始）,
  两职合一且两侧无同步约定 = 静默炸弹
- **决定性证据**: 同任务两次 create 只差后缀——带后缀空工作区 vs 原始完整仓库
- **解决**: Mac bundle 查找加 `-[0-9a-f]{12}` 剥离回退 + 回归测试钉死后缀格式
- **详细**: `GRPO_卡点沉淀_空工作区_task_id后缀.md`

### B2. 并发 env 共享竞态（v9 全 env_error）
- **现象**: 4/4 rollout 全灭 `term=env_error turns=3-4`——同 task 并发重叠实锤
- **根因**: server `envs[task_id]`（后到覆盖先到, 任一 destroy pop 后其余 404）
  + mock workspace 按 `repo-task_id` 键控（destroy rmtree 删目录 → 500）
- **解决**: `env_id = task_id-uuid8` 唯一化（**TaskSpec.task_id 原样不动**——B1 教训:
  只动 env_id 不动 task_id）+ mock workspace uuid 隔离 + Mac 同步重启
- **详细**: `GRPO_卡点沉淀_advantage恒0_预算截断finish决策点.md`（v9-v10 节）

### B3. 观测 AttributeError 误判 env_error（v11）
- **现象**: `err=execute str_replace failed: 'StrReplaceAction' object has no attribute 'arguments'`
  → 被外层 except 捕获 → 误判 ENV_ERROR → reward 0; Mac 侧零痕迹（响应后本地炸）
- **根因**: 观测代码写 `action.arguments.get('path')`, flat ToolAction 无 `arguments` 属性
  （parse 层已转 flat, 字段在 action.path 上）——观测层异常被当成环境错误
- **解决**: 观测 try/except 静默（观测职责是描述, 不是判定）
- **教训（铁律）**: **观测代码绝不引入异常路径**——观测失败把正常业务失败变成 reward 0

### B4. 观测截断保尾部（"LLM 看不到 Docker 报错"）
- **现象**: 7 条 rollout 3 条尝试编辑全失败（old_string not found ×1 + 绝对路径 ×2,
  其中一条用同一错误重试 26 次）——用户假设模型看不到报错, 部分成立
- **根因**: 渲染层 ✓ / 回灌层 ✓ / **截断层 ✗**: `observation_ids[:remaining]` 前缀截断,
  预算不足时末尾错误被整条切掉, 预算耗尽时整轮零反馈
- **解决**: `observation_ids[-remaining:]` 保尾部（注意 `-0` 陷阱先判 remaining>0）
- **教训**: 观测链路三层（渲染/回灌/截断）都要验证"错误消息可达模型"

---

## C. 模型输出协议侧（rollout 协议层）——3 个卡点

> 特征：模型文本 ↔ 动作的纪律问题——多调用、乱码、截断、不 finish、token 污染。
> 改的是 parse.py / verl_agent_loop.py / 采样与停止配置。根因常是**训练与推理不一致**。

### C1. SFT 格式 vs rollout 协议差距（行为层根因）
- **现象**: v2 8 步三恒 0（rewards/pg_loss/grad_norm）, entropy 非 0（policy 有效但零更新）
- **诊断链**: probe1（轮 256）一轮塞 2-3 调用 + JSON 截断 → PARSE_ERROR; probe2（预算 2048）
  **预算不是根因**（turn 2 纯泰文/阿拉伯文乱码 = 温度 1.0 采样发散）
- **数据断层假设被推翻**: `messages` 层 0 tool_calls 是展示层重建丢结构; **token 层**
  decode response_ids = 100% 含 JSON、100% 含 finish——模型学过格式, 是 rollout 复现不了
- **五处差距**: harness 只消费第一个调用 / 裸 JSON 无 `<tool_call>` / 温度 1.0 发散 /
  轮预算截断 / finish 永不触发
- **解决**: `parse_all_tool_calls` 多调用消费 + 温度 1.0→0.7 + 轮预算 512
- **详细**: `GRPO_卡点沉淀_奖励恒0_SFT格式与rollout差距.md`

### C2. 训练/推理停止条件不一致 → 洪水（v8）
- **触发**: 用户提问"SFT 是否没训练好只生成工具调用的能力？数据没组织好？"——
  **数据是好的, 停止条件不一致**
- **根因**: SFT 学"叙述 + 恰好 1 个 JSON + `<|im_end|>`"（im_end 是 mask=1 的 CE 目标,
  id=151645）; RL 侧 verl agent 路径 SamplingParams 无 stop 配置（vllm_async_server.py:505
  只透传 temperature/max_tokens）→ vLLM 在 im_end 后**继续生成** → 训练外位置自创续写 →
  一轮 8 个调用 = 洪水
- **解决**: 每轮 generate 注入 `stop_token_ids=[151645]` → 完全回到 SFT 消息节奏
- **教训**: 洪水类"模型行为异常"先查训练/推理的消息边界一致性, 再怀疑数据

### C3. 多调用 token 污染 policy loss（v23/v24 单调用协议）
- **v23**: 执行层只跑第一个调用, 但 generated_ids 全量写入 response_ids/mask=1/running_ids
  ——被丢弃的第二、三个调用仍参与 GRPO policy loss 并进入下一轮上下文
- **v24 三修复**: im_end 判定基于**截断后**保留序列（v20 stop_reason 豁免作废）/
  parse_error **不丢弃**全量写入（负奖励作用于错误 token）/ 候选按文本实际起始位置取最早
- **详细**: `GRPO_v24_核心修改与卡点沉淀.md`

---

## D. 训练/算法侧（veRL + reward + 显存）——6 个卡点

> 特征：reward 公式、advantage 机制、预算/激励/轮数三角、显存。改的是配置与奖励常数。

### D1. 预算截断在 finish 决策点之前（v4/v5）
- **现象**: 48 条 rollout 零 finish+patch → reward 恒 0.05（format 台阶人人有份）→
  advantage 恒 0 → pg_loss 0 → grad_norm 0
- **根因**: SFT 平均 **15.8 轮**才 finish; 预算 2048≈3.75 轮 / 4096≈6.5 轮全被 RESPONSE_CAP
  截死——模型不是不会 finish, 是**来不及 finish**
- **解决（v6）**: 预算 12288 + turns 24 + 逐条 `[rollout]` trace（终止可见性）

### D2. "到了也不 finish"——激励不足（v6 trace 证伪预算假设）
- **trace 实录**: 22 轮 / 144 调用 / 12288 tokens 仍零 finish → 预算假设死亡
- **附带发现**: 预算被多调用洪水 + 观察文本吃掉（aiohttp 每轮 8 调用, obs 回灌 71%）
- **解决（v7）**: finish 激励 **3:1**（finish+patch 0.15 vs 刷工具 0.05）+ 每轮调用封顶 4
  + 预算 9216 + 工具 mix trace

### D3. 纯二进制全 unresolved → 组内全同 → partial_reward 演进
- **v3 实测**: step 1 16/16 reward 0 → 纯二进制组内全同 → advantage 0 空转
- **v1**: `partial_reward`——resolved 锚 1.0; 未解决时按逐条 F2P 真实通过给部分分 +
  P2P 破坏惩罚（防刷分）+ format 小台阶; 每分来自 CleanVerifier 真实 exit code, 不伪造
- **v17 step1 实证又卡**: patch 2/8 但 f2p 全 0 → reward 恒 0.05 → 零梯度
- **v4**: patch 台阶 +0.10（仅 f2p_credit==0 时给）+ clean/finish 台阶 + parse_error 惩罚
  → 制造 0.05 vs 0.15+ 方差
- **验证（v24 run5 step2）**: advantages max +1.42 / min -1.50——梯度在流动

### D4. 显存翻车 1: v6 entropy 翻爆
- 预算 12288 → ppo_max_token_len_per_gpu=15360 → entropy softmax 15360×151,936 logits
  ≈18GB 临时张量（**预算翻倍把熵计算翻爆**, 非权重/激活）
- 解决: 预算 9216 + 两点线性外推显存表

### D5. 显存翻车 2: v12 wake_up cumem OOM
- update 后 vLLM wake_up 需引擎池 6.0GB; 估算漏了它（PyTorch caching allocator 保留
  峰值不归还, 空闲 1.9GB < 6.0GB）
- 解决: 预算 9216→6144 + GPU_MEM_UTIL 0.25→0.20 + max_num_seqs 6→5

### D6. 显存翻车 3: v24 run2/3 obs 累积打满
- turns 24 → obs 累积 update 序列 ~24000 → FSDP 激活 24.5GB 全满 → wake_up 无空间
- 解决: turns 24→12 + `ppo_mini_batch_size` 4→1（激活减半）+ 大预算 6144 不砍
  （v24 step1 实测峰值 19.2GB, 余量 5.4GB）
- 详细: `GRPO_v24_核心修改与卡点沉淀.md` §2

---

## E. 数据侧（SFT/pool 输入原料）——2 个卡点

> 特征：输入原料有坏内容/太难。改的是数据管线与任务池。

### E1. r2egym 路径泄漏（/testbed 绝对路径进 SFT 目标）
- SFT 数据里工具路径含 `/testbed` 绝对路径 → 模型学到错误路径模式（rollout 是相对路径）
- 解决: normalize 修正为相对路径 + 回归测试 + 重跑 SFT 验证泄漏清零
- 详细: `SFT_路径泄漏修复与遮挡说明_20260811.md`

### E2. 任务池难度超模型能力（v4-v22 长期）
- 142 条 Mixed Pool（aiohttp/humanize/dateutil/toolz/more_itertools）真实库任务,
  4B 修复概率存疑——"模型能力 vs 任务难度"错配
- 解决（v23）: 降难度池（30 AST + 50 Reversal 优先池）+ priority_pool 采样权重
  （easy/unsolved 恒 0 不占采样）

---

## F. 模型能力侧（能力边界, 非 bug）——2 项

> 特征：链路修好后仍然存在的行为, 属于模型真实能力, 靠 RL 正向奖励流解决, 不修 harness。

### F1. 0.6B 未学会 tool_call 语法
- 阶段 17 记录 reward mean=0 归因模型能力——与 C1 同根因家族, 但 0.6B 是纯能力不足
  （4B+SFT 是泛化与采样问题）

### F2. 4B 无视可见错误重试 26 次
- 观测截断修复后（B4）错误已可达, 但模型仍重复相同错误——早期 RL 无正向奖励流
  时模型"学不会看错误"; 需要奖励信号引导

---

## 跨侧卡点图（一个卡点, 两侧修复）

```
B1 task_id 后缀: 根因在 B（键两职合一）→ 修复在 A 侧（Mac bundle 剥离回退）
A2 str_replace:  根因在 A（执行器无兜底）→ 修复跨 B（文件错误应转 observation 的语义规则）
C1 协议差距:     根因在 C（rollout 复现不了 SFT 格式）→ 修复跨 B（harness 多调用消费）
E1 路径泄漏:     根因在 E（数据）→ 若只修 harness 会掩盖, 必须改数据 + 重训
```

**判定规则**: 症状全在 reward 数值上, 但改哪一侧由根因层决定——先对照探针定位根因层,
再决定改 Mac（A）/契约（B）/parse-loop（C）/配置-奖励（D）/数据（E）/不修（F）。

---

## 版本演进对照（v1 → v24）

| 版本 | 侧 | 关键动作 |
|---|---|---|
| v1 | A | bundle 缺失 → 部署 + gold 探针 |
| v2 | B | reward 恒 0（task_id 后缀, 未修） |
| v3 | D | partial_reward v1（F2P partial） |
| v4 | D | 预算 2048 → uniform 0.05（预算截断） |
| v5 | D | 预算 4096 → 同 uniform |
| v6 | D+C | 预算 12288 + turns 24 + trace → 证伪预算假设; entropy OOM |
| v7 | D+C | finish 3:1 + 调用封顶 4 + 预算 9216 |
| v8 | C | stop_token_ids im_end（停止条件一致） |
| v9 | B | env_error 全灭（并发 env 共享竞态） |
| v10 | B | env_id 唯一化（server.py + mock.py） |
| v11 | B | 观测 AttributeError 误判 |
| v12 | B+D | 观测静默化; wake_up OOM → 预算 6144 |
| v13 | B | 观测截断保尾部 |
| v14-15 | D | 塑形 v2/v3（finish 台阶等） |
| v16 | E | 纯 R2E 池（用户指示） |
| v17 | B+D | observation 归一化 + prompt v3; patch 2/8 但 f2p 0 → 零梯度实证 |
| v18 | D | 塑形 v4（patch/clean/finish 台阶 + parse_error 惩罚） |
| v19 | C | view 截断指引 + prompt v4（已停, 实证失败） |
| v20 | C | prompt v5 真实 im_end fewshot |
| v23 | C+E | 单调用协议 + 降难度池（30 AST + 50 Reversal） |
| v24 | C+D | 三阻塞修复 + turns 12 + mini_batch 1 + 大预算 6144 |

---

## 卡点模式总结（按侧）

**A 侧**: 执行器必须兜底一切文件系统场景（目录缺失/文件缺失都是 observation 不是 500）
**B 侧**: ① 两职合一的 key 是静默炸弹, 必须两侧同步约定 + 回归测试钉死 ② 观测代码绝不
引入异常路径 ③ 观测链路三层（渲染/回灌/截断）都要验证"错误消息可达模型" ④ "每环都健康"
是最大迷雾, 对照探针是唯一分辨手段
**C 侧**: ① 训练/推理停止条件必须一致（im_end、单调用节奏）② 判断模型学到了什么看 token
层, 不看 messages 层 ③ 响应预算是上限不是消费目标, 但不能低于模型实际消耗（v24 实测
mean 4086, 2048 3 轮截死; 参考手册 ≠ 照抄）
**D 侧**: ① 组内方差是 GRPO 命脉——reward 全同 → advantage 0 空转; 方差靠 partial_reward
台阶 + 组内多样性制造 ② 预算/激励/轮数三角: 来不及 finish / 到了也不 finish（激励 3:1）/
轮数上限先于预算截死 ③ 显存估算必须含全部阶段（entropy 临时张量、wake_up 引擎池、
obs 累积进 update 序列）
**E 侧**: 输入原料先于 harness 修（路径泄漏不改 harness 掩盖）
**F 侧**: 能力边界不修 harness, 靠 RL 正向奖励流

---

## 文档索引

| 文档 | 侧 | 一句话 |
|---|---|---|
| **本文件** | 全 | 按系统侧分类的全部卡点 + 归属判定 |
| `GRPO_卡点沉淀_Docker环境.md` | A | bundle 缺失/数据质量 |
| `SWE-Forge_formal任务Docker空工作区问题复盘.md` | A 复盘 | bundle 未部署 |
| `GRPO_卡点沉淀_空工作区_task_id后缀.md` | B ★ | 后缀打爆 bundle 查找（结论翻转） |
| `GRPO_卡点沉淀_Mac_str_replace_mkdir.md` | A+B | 目录缺失 500 修复 |
| `GRPO_卡点沉淀_奖励恒0_SFT格式与rollout差距.md` | C+D+E | 协议差距 + partial_reward 演进 + 路径泄漏 |
| `GRPO_卡点沉淀_advantage恒0_预算截断finish决策点.md` | B+C+D | v4→v13 全链（预算/激励/停止条件/env_error/观测/wake_up） |
| `GRPO_v24_核心修改与卡点沉淀.md` | C+D | 三阻塞修复 + 显存峰值方案（run5 实证） |
| `SFT_路径泄漏修复与遮挡说明_20260811.md` | E | /testbed 泄漏修复 |
| `GRPO_操作手册.md` | 操作 | run5 参数、验证点、故障表 |
| `GRPO_正式运行手册.md` | 操作（旧） | v4 时代手册（配置已被 v24 取代） |
