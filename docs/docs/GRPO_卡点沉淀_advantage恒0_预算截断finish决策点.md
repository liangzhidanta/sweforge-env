# GRPO 卡点沉淀：advantage 恒 0 死局——预算截断在 finish 决策点之前（2026-08-11）

> 目标：v4/v5 正式 GRPO"reward 全 uniform → advantage 恒 0 → pg_loss 恒 0 →
> 权重不动"的完整诊断与 v6 修复。这是"模型能力层"之前的**信号层死锁**：
> 奖励函数设计得再好，模型到不了任何可区分的终点，梯度就永远不出现。

## 现象（日志证据）

- **v4 steps 1-3 + v5 steps 1-2 = 连续 5 个 uniform step**：
  - v5 step 1: `critic/rewards/mean=min=max=0.05` → `advantages 全 0` → `pg_loss=0` → `grad_norm=0`
  - v5 step 2: 同样全 0.05 / 全 0（16 条 rollout 逐条全同）
- **48 条 rollout（v4 32 + v5 16）零 finish+patch、零 F2P credit**：
  reward 恒 = format 台阶 0.05（有 ≥1 个工具调用的人人有份，无区分度）。
- 唯一出现过的一次方差：v4 step 2 组 {0.05, 0.05, 0.05, **0.0**} →
  advantage 精确 +0.5/−1.5（GRPO 样本 std 归一化）。那 0.0 是 parse-error
  （无工具调用）——**噪声梯度，不是学习信号**（方向 = "别解析失败"）。

## 根因（信号层死锁的因果链）

1. **GRPO 组内全同 → std=0 → advantage=0 → pg_loss=0**：
   `advantage = (r − group_mean) / sample_std(n−1)`；reward 全同则恒 0。
2. **模型到不了 finish 决策点**：
   - SFT 轨迹平均 **15.8 轮 / 10,000+ chars** 才 finish（2904/2904 条轨迹 100%
     以 finish 结尾——finish 行为 SFT **已学会**）。
   - v4 预算 2048 token ≈ **3.75 轮**；v5 预算 4096 ≈ **6.5 轮**（num_turns
     实测 6.5）→ 全被 RESPONSE_CAP 截死，从未到过模型学到的
     "探索够了，该 finish 了" 的决策点。
   - 即：模型在 RL 里不是"不会 finish"，是"**来不及 finish**"。
3. **format 台阶饱和**：+0.05 对"有工具调用"人人有份（v5 prompt 契约后
   16/16 都有）→ 组内无区分 → 连 v4 那种 parse-error 0.0 方差都消失了。
4. **逐条可见性缺失**：只有聚合 reward，分不清"预算不够没来得及" vs
   "给了预算也不 finish" —— 诊断全靠猜。

## 修复（v6 三件套）

1. **response 预算 4096 → 8192 → 12288**（≈24 轮，覆盖 SFT 15.8 轮均值 +
   尾部）。预算是**上限不是消费目标**：让模型自己决定何时 finish。
2. **max_assistant_turns 12 → 24**（`configs/agent_loops.yaml`）——轮数上限
   必须先于预算截死，否则预算白给。
3. **逐条 [rollout] 终止 trace**（`verl_agent_loop.py` 收尾打印进 train.log，
   可观测性，不动 verl/不改 reward）：
   `[rollout] task=… term=finish turns=7 tools=9 parse_err=0 resp=6234 score=0.10 patch=312 f2p=1/2`
   → `grep "[rollout]" train.log` 即得逐条行为，决策不再靠猜。

配套显存调整（TP=4, KV 36KB/token/card 实测标定）：
- `GPU_MEM_UTIL 0.2 → 0.25`（vLLM 预算 4.8 → 6.0GB，KV 可用 2.8 → 4.0GB；
  每卡基线 40% 实测，u=0.25 后 vLLM ~6GB + FSDP ~7GB ≈ 13GB 稳态仍在 24GB 内）。
- `max_model_len = 3072+12288 = 15360`；`max_num_seqs 8 → 6`：
  KV = 6×15360×36KB = **3.32GB ≤ 4.0GB**（0.68GB 余量；装不下在 vLLM
  启动时直接 fail-fast，不污染 run）。

## 验收标准（v6 steps 1-3）

- [ ] `[rollout]` trace 出现 **term=finish**（模型自己走到 finish 决策点）
- [ ] `critic/rewards` 出现组内方差（finish+patch 0.10 / F2P credit > 0.05）
- [ ] `critic/advantages` 非零 → `pg_loss < 0` → `grad_norm > 0`（梯度流动）
- 若 3 步后仍 uniform：trace 区分"预算不够"（resp 打满仍无 finish）
  vs "模型不 finish"（resp 未满就 MAX_STEPS）——前者加预算，后者转
  提示词/任务难度杠杆（**禁止改 reward 公式**，P5）。

## 附：GPU 利用率对比（v4 空闲块 vs v5 连续）

用户观察：v4 每步有一整段 GPU 0% 利用率；v5 连续波动无整块空闲。原因：
- v4：`max_num_seqs=4` + 16 rollout/step → 同任务同 prompt 的 4 条并发
  rollout 行为同步 → **同时卡在 Mac Docker 往返** → GPU 整块等远端。
- v5：`max_num_seqs=8` + 2 任务错峰 → 等 Mac 的同时别的 rollout 在生成
  → GPU 连续利用（generate_sequences 160s → 132s 印证吞吐改善）。

## 铁律合规

- 未改 verl；未改 reward 公式（format/finish 台阶保持 v2/v3 语义）；
  trace 是纯打印；预算/轮数是配置；reward 每分仍来自 CleanVerifier 真实执行。

---

# 追加：v6 崩溃与 v7/v8 修复（2026-08-11）

## v6 OOM（torch.OutOfMemoryError, update_actor 前 forward 阶段）

- **崩溃点**: `prepare_model_outputs -> entropy_from_logits -> softmax` 尝试分配
  7.27 GiB 失败（进程已用 16.05 GiB, 卡 23.52 GiB）。
- **根因**: response 12288 -> ppo_max_token_len_per_gpu = 3072+12288 = **15360**
  -> dynamic bsz 按 15360 token 打包 micro-batch -> entropy 在
  15360×151,936 logits 上 softmax ≈ **18GB 临时张量** + FSDP 权重 ~7GB。
  v5（chunk 7168）峰值 13.7GB 不爆; **预算翻倍把熵计算翻爆**——显存风险
  随 budget 线性放大, 不是权重/激活问题。

## v6 trace 决定性发现（预算假设被证伪）

v6 崩溃前 8 条 rollout 逐条（12288 预算 / 24 轮上限 / u=0.25）：

```
turns=7  tools=56  resp=12288  f2p=0/1   <- aiohttp: 一轮 8 个调用洪水
turns=8  tools=64  resp=12288  f2p=0/1
turns=18 tools=142 resp=12288  f2p=0/1   <- reversal: 144 调用仍零修复
turns=22 tools=52  resp=12288  f2p=0/1   <- 22 轮仍不 finish
```

1. **"预算不够"假设死亡**: 22 轮 / 144 调用 / 12288 tokens 全给出, 模型仍
   零 finish、零 f2p 翻转。不是"来不及 finish", 是"**到了也不 finish**"。
2. **预算被"多调用洪水 + 观察文本"吃掉, 不是被探索深度吃掉**（用户观察
   独立验证）: aiohttp 每轮 8 调用（512 token 模型续写里塞 6-8 个 JSON）,
   观察回灌 71% 预算; reversal 144 调用本身吃 75% 预算。
3. **zero f2p 翻转 = 能力/激励双缺失**: 144 调用后 F2P 测试仍失败——模型
   从没把测试变绿（可能根本没跑 pytest, 工具 mix 当时不可见）。

## v7 修复（三件套 + 一项观测）

1. **finish 激励 3:1**（reward/verifier.py, P5 合规——调 shaping 常数非伪造）:
   v6 时 finish+patch 0.10 vs 刷工具 0.05 只差 2:1, 继续刷无惩罚、收尾无
   收益, 模型刷满预算也不收尾。FINISH_WEIGHT 0.05→**0.10**（finish+patch
   = 0.15 = 3× 刷工具）, F2P_MAX 0.90→**0.85**（常量之和保持 1.0, 未解决
   严格 <1.0 不变式由"f2p_credit<1 ∨ p2p_factor<1 ⇒ score<1"结构性保持）。
2. **每轮调用封顶 8→4**（verl_agent_loop.py parse_all_tool_calls max_calls）:
   system prompt 明令 "exactly one tool call per response" 但 4B 分布外
   不遵守——parse 层强制拉回 SFT "每消息 1 调用"节奏（SFT 100% finish
   发生在这种节奏里）。观察成本减半, 预算花在更多有反馈的轮次。
   尾部残缺 JSON 由 parse 层跳过（宽容策略不变, 不产生 ParseError）。
3. **显存安全**: response 12288→**9216**（chunk 12288, 熵峰值估计 ~19GB,
   v5 7168→13.7GB 与 v6 15360→23.3GB 两点线性外推, 留 ~5GB 余量）。
   调用封顶后每轮消耗降 ~40%, 9216 ≈ 18 轮有效轮次, 与 SFT 15.8 轮均值匹配。
4. **trace 加工具类型 mix**: `tools=56 [bash8/srch20/view22/edit6]` —— 下轮
   直接可见"模型在跑 pytest/编辑 vs 纯 view 洪水", 判断能力缺口无需再猜。

## v7 验收标准（steps 1-3）

- [ ] `[rollout]` trace 出现 **term=finish**（finish 激励生效）
- [ ] `critic/rewards` 组内方差（0.05 / 0.15 / f2p credit 组合）
- [ ] `critic/advantages` 非零 -> `pg_loss < 0` -> `grad_norm > 0`
- [ ] 工具 mix 显示 str_replace/bash(pytest) 存在且占比合理
- 若 3 步后仍 uniform：剩余杠杆 = **任务难度课程**（当前 pool 全为真实
  库 aiohttp/humanize/dateutil/toolz/more_itertools, 4B 修复概率存疑;
  可选生成 demo_calc toy 池 + Mac bundle 同步）或提高采样温度/组内方差。

## 洪水根因（v8, 停止信号缺失——训练/推理不一致）

用户提问触发深挖: "SFT 是否没训练好根据历史会话只生成工具调用的能力？
是不是数据没组织好？" —— **数据是好的, 停止条件不一致**:

1. **SFT 侧**: `data/sft/tokenize.py` 逐条消息 `apply_chat_template` →
   assistant 消息 = `<|im_start|>assistant\n` + [叙述 + **恰好 1 个** JSON] +
   `<|im_end|>`（`assistant_prefix_mask` 只剥前缀, **im_end 是 mask=1 的
   CE 学习目标**）。模型学的就是"JSON 写完 → im_end 收尾, 每消息 1 调用"。
   证据: Qwen3-4B tokenizer 渲染 assistant 消息尾部 = `</tool_call><|im_end|>`,
   im_end id=151645。
2. **RL 侧**: `verl_agent_loop.py` prompt 同样 apply_chat_template（上下文
   一致）-> 模型按学到的分布生成"叙述 + JSON + im_end"（~100-200 token
   就该停）-> 但 verl agent 路径 SamplingParams 无 stop 配置
   （`vllm_async_server.py:505` 只透传 temperature/max_tokens）-> vLLM 在
   im_end 之后**继续生成** -> 模型被顶到"消息结束、观察未至"的**训练外
   位置** -> 按"开新消息"模式续写 JSON -> 循环填满 512 token 预算 ->
   parse_all_tool_calls 提取全部完整 JSON -> 全部执行 -> 一轮 8 个调用
   = **洪水**。模型不是"学会了狂写", 是"正确地结束消息而 harness 无视
   结束信号"。
3. **修复（v8, loop 内 3 行, 不改 verl）**: 每轮 generate 注入
   `stop_token_ids=[151645]` -> vLLM 在 im_end 处停 -> 每轮 1 调用 +
   观察回灌, 完全回到 SFT 消息节奏; max_calls=4 保留作安全阀（模型从不
   生成 im_end 时洪水仍被封顶）; trace 加 `clean=n/turns`（以 im_end 收尾
   的轮数 = 停止信号兑现率, 预期接近 1.0）。

教训: 洪水类"模型行为异常"先查**训练与推理的消息边界/停止条件是否一致**,
再怀疑数据组织——CE 教的是条件分布, 推理时兑现不了停止信号, 模型只能在
训练分布外自创续写。

## 显存外推表（供后续预算决策）

| response | ppo chunk | entropy 峰值估计 | 实测/外推 |
|---|---|---|---|
| 4096 (v5) | 7168 | ~8.6GB | 实测峰值 13.7GB ✓ |
| 9216 (v7) | 12288 | ~14.7GB | 外推 ~19GB（预计安全） |
| 12288 (v6) | 15360 | ~18.4GB | 实测崩溃 23.3GB ✗ |


# 追加：v9 全 env_error —— 并发同 task env 共享竞态（2026-08-11）

## 现象（日志证据）

v9 启动后 4/4 rollout 全灭: `term=env_error turns=3-4`（工具混用正常
[search/view/str_replace/bash], 但第 3-4 轮 execute 抛异常, score 0.0）。
4 条 [rollout] 打印在连续 5 行内 = 同一时刻收尾 = 同 task 并发重叠实锤。

## 根因（双层共享, 两层都要修）

1. **server envs 注册表 key = task_id**（server.py:109-110）: GRPO
   `--batch 2 --n 4` = 同一 prompt（同一 task_id）采样 4 条并发 rollout →
   4 个并发 POST /v1/envs 全部写入 `envs[task_id]`, 后到者覆盖先到者 →
   全部 rollout 共享同一 env 实例。任一 rollout destroy →
   `envs.pop(task_id)` → 其余 rollout 的 execute/export_patch → **404
   env not found → env_error**。v8 未炸是 rollout 长（response_cap）,
   同 task 4 条时间错开躲过窗口; v9 im_end 停止让每轮变快、重叠必现。
2. **MockEnvironmentBackend workspace 按 `repo-task_id` 键控**（mock.py）
   : 同 task 并发 env 共享同一工作目录, 任一 destroy `rmtree` 删掉目录 →
   其余 env execute → **FileNotFoundError → 500**（本地复现实测）。
   Mac 侧 TaskEnvironmentBackend 用 mkdtemp 独立目录（不受此层影响）。

CLAUDE.md 声称 "每并发 rollout 唯一 env instance（uuid 后缀）" —— 设计
意图如此, 实现从未落地（env_id 恒 = task_id）。本次修复兑现设计意图。

## 修复（v10, 纯 AutoDL 侧 + server 侧）

1. **server.py create_env**: `env_id = f"{task.task_id}-{uuid4().hex[:8]}"`,
   `envs[env_id] = backend.create(task)` —— 每条 rollout 独立 env。
   **TaskSpec.task_id 原样不动**（#86 教训: 只动 env_id, 不动 task_id,
   bundle 查找/verify 不受影响）。reset/actions/patch/destroy 端点按
   path env_id 查 envs, 无需改。destroy 404 幂等已容错。
2. **mock.py create**: workspace 加 uuid 后缀逐 env 隔离（否则回归测试
   "destroy A 不影响 B" 永远过不了——这正是测试抓到的第二层问题）。
3. **测试**: test_environment_server 三处 env_id==task_id 断言改为
   "task_id 前缀 + 唯一 + destroy A 后 B 仍可执行"（竞态回归断言）。
   全量回归 424 passed。

## 部署要求（Mac 侧必须同步重启）

server.py 改动在 repo; Mac 上跑旧代码 = 训练仍会 404。需要:
`cd <Mac repo> && git pull && 重启 server 进程`（同一命令行/服务方式）。
重启后 curl /health 验证。**训练期间不要重启 server**（铁律）。

## 验收标准（v10 steps 1-3）

- [ ] `[rollout]` trace 无 env_error（同 task 并发不再互杀）
- [ ] `term=finish` + 非空 patch（finish 激励 3:1 生效）
- [ ] `critic/rewards` 组内方差 -> advantage != 0 -> pg_loss < 0 -> grad_norm > 0
- [ ] entropy 分块后 step 1 update 不再 OOM

---

# v10 仍 2/3 env_error——真根因: 观测代码 AttributeError 误判（2026-08-11）

## 症状

v10（env_id 唯一化已部署, Mac server 零 404/500）仍 2/3 env_error。
Mac 侧 per-env 生命周期全量分析: 无 404/500/traceback、无 create 后
零 action 的 env、短命 env 是探测型。**服务器侧完全没有对应故障**。
候选收窄到: ① 客户端 execute 超时 ② 反向隧道丢包 ③ 客户端本地异常。

## 实锤（v11 err= trace, 第一条 rollout 即现行）

```
err=execute str_replace failed: 'StrReplaceAction' object has no attribute 'arguments'
```

完整链条:
1. 模型 str_replace 业务失败（old_string 拼错, success=False）——
   **服务器正常返回 observation**（零 404/500 的原因）
2. v9 加的观测代码渲染 editfail 详情写 `action.arguments.get('path', '?')`,
   flat ToolAction **无 `arguments` 属性**（模型 JSON 文本才是
   {name, arguments} 包装, parse 层已转 flat, 字段在 action.path 上）
   -> AttributeError
3. 被 rollout 外层 except 捕获 -> 误判 ENV_ERROR -> reward 0

## 三方印证

- Mac 零痕迹: 请求全部成功, 异常发生在响应到达之后的本地代码, 零请求发出
- v8 零 env_error / v9-v10 全灭: 观测代码 v9 才加
- turns 3-4 挂: 模型先探测几轮, 第一次 str_replace 失败即触发

## 排除项（本轮明确排除, 不用再查）

- 容器复用: Mac 代码级确认 create 无复用逻辑, 每次 uuid 后缀新容器
- 隧道丢包 / execute 超时: 若发生会留 404/500 或请求缺失, 且报错文案
  不会是 AttributeError

## 修复（v12, 双保险）

```python
try:
    if (action.name == "str_replace" and str_replace_fail is None
            and isinstance(getattr(observation, "success", None), bool)
            and not observation.success):
        reason = getattr(observation, "error", None) or "no error detail"
        path = getattr(action, "path", None) or "?"
        str_replace_fail = f"{path}: {reason}"[:200]
except Exception:
    pass  # 观测失败静默（观测绝不当环境错误）
```

回归: test_str_replace_business_failure_does_not_terminate（业务失败
不终止 rollout、正常 finish、destroy 清理）。

## 教训（写进铁律精神）

**观测代码绝不引入异常路径**——观测失败必须静默（try/except pass），
观测职责是描述, 不是判定。任何观测层异常被误判为环境错误 = 把正常
业务失败变成 reward 0, 直接污染 RL 信号。

## v12 验收

- [ ] `[rollout]` trace 无 env_error
- [ ] `term=finish` + 非空 patch; `editfail=` 显示 str_replace 失败原因
      （观测正常工作的证据, 可与 Mac server log 对拍）
- [ ] critic/rewards 组内方差 -> advantage != 0 -> pg_loss < 0 -> grad_norm > 0
- [ ] entropy 分块后 step 1 update 不再 OOM

---

# v12 step 1 wake_up OOM——显存实测推翻估算（2026-08-11）

## 症状

v12（env_error 修复版）第一次完整跑完 rollout + step 1 update,
update 结束后 vLLM wake_up() 抛 `CUDA Error: out of memory at
cumem_allocator.cpp:62`。v8-v11 从未到此: v9/v10 全 env_error、
v11 未跑完——**这是第一次 step 1 update 活着跑完**。

## 实测数据（system.jsonl）

- rollout 阶段: 11.3 GiB 稳定（vLLM 引擎池 6.0GiB + FSDP 权重）
- update 阶段: 9.7 -> 13.2 -> 17.7 -> 19.7 -> **21.6 GiB 峰值**
- wake_up 时刻: PyTorch caching allocator 保留峰值不归还驱动 ->
  空闲 23.5 - 21.6 = 1.9 GiB < vLLM 引擎池 6.0 GiB -> 失败

v9 文档"12288 chunk 峰值 ~21.5GB 留 2GB 余量"的估算漏了 wake_up 的
6GB 引擎池——估算 vs 实测的经典教训。

## 修复（v13, 实测点线性外推）

两个实测点: v5 chunk 7168 -> 13.7GB; v12 chunk 12288 -> 21.6GB。
斜率 ~7.9GB/5120 token。

1. response 9216 -> 6144（chunk 12288 -> 9216, 峰值估 ~16.9GB）
2. GPU_MEM_UTIL 0.25 -> 0.20（引擎池 6.0 -> 4.8GB）
3. max_num_seqs 6 -> 5（KV 2.65 -> 2.21GB <= 4.8GB 池）
   wake 需求 = 16.9 + 4.8 = 21.7GB, 余量 ~1.8GB。

代价: 单 rollout 探索预算 -33%（模型 0/7 finish 烧满 9216 也白烧,
稳定性优先; 学到 finish 后再考虑加回）。

---

# 观测截断保尾部——"LLM 看不到 Docker 报错"（2026-08-11, 用户提问驱动）

## 问题

7 条 rollout 中 3 条尝试编辑全失败（old_string not found ×1 +
绝对路径 path must be relative ×2, 其中一条用同一错误重试 26 次）。
用户假设: 模型是否看不到 Docker 返回的报错?

## 验证（部分成立）

- 渲染层 ✓: serialization.py render_observation 对 str_replace 失败
  渲染 `[str_replace failed] {error}`——错误 100% 进文本
- 回灌层 ✓: obs_wires 全部编码进 response_ids -> running_ids -> 下一轮
  context, 模型应该看到
- **截断层 ✗（真漏洞）**: `observation_ids[:max(0, remaining)]` 前缀
  截断——一轮 4 个调用时 wire = [obs1][obs2][obs3][obs4 错误], 预算
  不足时先到的 search/view 保住, **末尾错误被整条切掉**; 预算耗尽时
  整条 wire 归零, 模型最后一轮零反馈

## 修复（v13）

`observation_ids[-remaining:]` 保尾部（注意 -0 陷阱: 先判 remaining>0）。
牺牲开头 search/view 长文本, 保住末尾错误消息。回归测试
test_observation_truncation_keeps_tail_error: 同一轮 bash 长输出 +
str_replace 失败, 截断后 "old_string missing" 在、"exit code" 不在。

## 教训

- 观测链路三层（渲染/回灌/截断）都要验证"错误消息可达模型"——
  渲染和回灌正确不代表最终进 context
- 26 次重复的主因仍是 4B 无视可见错误（早期错误可见仍重试）,
  截断漏洞是加剧因素; RL 学会看错误需要正向奖励流
