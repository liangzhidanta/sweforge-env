# GRPO 卡点沉淀：reward 恒 0 的 Docker 环境根因（正式任务 bundles 缺失）

> 版本：2026-08-10。沉淀对象：formal Agentic GRPO（250 任务池 × Mac Docker CleanVerifier 真实验证）。
> 姊妹篇：`docs/GRPO_卡点沉淀.md`（OOM / 多卡适配 / verl 边界）。遇到"reward 恒 0 / advantage 恒 0 / pg_loss 恒 0"先读本文档。

---

## 一、现象：reward / advantage / pg_loss 三恒 0

- formal 池 GRPO 训练（阶段② 稳定性验证, v1 与 v2 两轮, 每轮 5 step）：
  - `reward/mean` = 0.0、`advantages/mean` = 0.0、`pg_loss` = 0.0——**完全没有学习信号**。
  - 训练本身"健康"：无 OOM（FSDP 峰值 10.17→11.47GB）、`update_weights` 正常 8.1s、工具调用真实到达 Mac Docker（ToolAction 执行成功, 有观测返回）、干净退出。
- 模型并非不干活：v1 时 `num_turns/mean=5.9`（模型在容器里反复试错）, v2 时 2.69（描述更具体后更快收敛到"没法修"的结论）。
- 0.6B demo 联调历史上拿过 `resolved=1.0`（阶段 18 E2E, 真实 Mac Docker）→ 排除"模型太弱必得 0"的简单归因。

## 二、错误归因 vs 正确归因

| 阶段 | 归因 | 为什么错 / 对 |
|---|---|---|
| 前期（错） | "reward 函数不合理：binary 太严, 应该给过程分" | reward 对真实测试语义正确（空 patch→0, demo 真实修复→1.0）。改 reward 是无用功 |
| 前期（错） | "0.6B 没学会 tool_call 语法, 所以恒 0" | demo 池同模型同语法拿过 1.0；v2 描述更具体后模型行为明显变化 |
| 中期（半对） | "problem_statement 只有 3 句模板, 零线索" | 真实缺陷但非根因：v2 丰富化 250 条具体描述（文件:行+失败测试名）后**依旧恒 0** |
| 正确 | **Mac Docker 容器里根本没有任务代码**：`/workspace` 只有空 git（1 个 baseline commit）+ .gitignore；`src/`、`tests/` 全不存在 | 测试全部 `file or directory not found` 假失败 → 任何 patch 都通不过 → reward 恒 0 是**环境假象** |

## 三、诊断方法（可复用：把"模型不行"与"环境不行"分开）

核心思路：**用"已知正确修复"（gold patch）当正例对照，探环境；用空 patch 当负例对照，探判据。** 探针脚本 `scripts/probe_gold_patch_reward.py`：

```bash
PYTHONPATH=src python scripts/probe_gold_patch_reward.py
```

对每个 repo 抽 1 条任务（hum-v5 / toolz-pre2 / du-v2 三条）走真实链路：

| 探针 | 期望 | 实测（2026-08-10） | 结论 |
|---|---|---|---|
| 空 patch | `unresolved, reward=0.0`（F2P 失败） | 完全符合 | verifier 判据语义正确 |
| gold patch | `resolved, reward=1.0`（F2P 全过+P2P 全过+integrity） | **`integrity=False` + P2P 全 False**；测试 stderr 显示 `ERROR: file or directory not found: tests/test_filesize.py::test_naturalsize` | **环境坏, 不是 reward 坏**——容器里连测试文件都不存在 |

随后直查容器（Mac 侧）：

```bash
docker exec <container> ls -la /workspace   # 只有 .git + .gitignore, 无 src/tests
```

证据链闭环：gold patch 是"必然正确"的修复, 它都失败且败在"文件不存在"上 → 与模型产出无关, 与 reward 公式无关 → 唯一解释是环境没给任务代码。

## 四、根因：bundle 同步是数据链路的隐式依赖

### 4.1 架构事实（本卡点的坑根）

```
AutoDL（训练侧）                     Mac（执行侧, 本 repo 看不到）
  pool.jsonl ──(HTTP §8 契约)──►   sweforge.env_server.server
   RemoteEnvironmentBackend          --bundles-dir <dir> --docker
        │                            LocalDocker backend
        └──── 8 端点唯一通道          task_id ──► <bundles-dir>/<task_id>/repo
       （无文件通道, 无 shell）         ├─ repo/                # base_commit 代码快照
                                      ├─ private/hidden_tests/ # Mac 侧注入测试
                                      └─ task_manifest.json     # image/workspace/seed_from_snapshot
```

- **Mac 侧实现（`--bundles-dir` / LocalDocker / bundle 查找）不在本 repo**：AutoDL 侧只有 §8 契约 server（`src/sweforge/environment/server.py:155-159` `/v1/verifications` 委托 `backend.verify`）。这意味着 AutoDL 侧**无法读 Mac 代码**, 只能靠运行时证据（探针输出、容器 ls）定位。
- bundle 协议由 `scripts/build_ast_bundles.py` 定义（manifest: `image`、`workspace=/workspace`、`seed_from_snapshot`、`integrity_protected` = f2p+p2p 测试文件名）——**构建 bundle 与同步 bundle 是两个动作, 前者不会自动送达后者**。
- formal 池的 250 条任务是在 AutoDL 侧工厂构建并 verified 的（测试在 AutoDL 侧临时 clone 里跑）；**Mac server 的 `--bundles-dir examples` 只有 demo bundles**, 没有任何一条 formal 任务 → 按 task_id 找不到 repo → 容器初始化回落到"空模板"（只有空 git）。

### 4.2 为什么容器里有个空 git（而不是报错）

容器初始化按 manifest `seed_from_snapshot` 语义执行：有 snapshot 就注入 base_commit 代码, 没有就只建空 git 骨架。于是：

- 五工具、观测、export_patch **全部正常**（架构层健康）→ 训练日志毫无异样；
- 只有跑测试那一瞬间暴露：`file or directory not found`。
- **reward 恒 0 是"正常运行的假环境"的必然产物**——链路里没有任何一环报错, 这就是它最难排查的原因。

## 五、解决方案：从工厂 work 仓库回捞代码快照

### 5.1 关键洞察

250 条任务的 `base_commit` 全部存在于**当初 factory 构建残留的 work 仓库**（未清理）：

```
formal/*/*/work/<repo>-work       # 13 个
preflight_v2/*/work/<repo>-work   # 2 个
```

直接 `git archive base_commit` 即可得到与 pool 精确一致的代码快照——**无需重建种子 / 重新 replay mutation**（阶段 16 的模板重放 hash 断言天然保证一致性）。

### 5.2 构建脚本 `scripts/build_grpo_bundles.py`

```bash
PYTHONPATH=src python scripts/build_grpo_bundles.py \
    --full <full.jsonl> --experiments <rl_formal_v1> --out <bundles_root>
```

- `discover_work_repos`：glob `formal/*/*/work/*-work` + `preflight_v2/*/work/*-work`
- `has_commit`：`git cat-file -t` 判定 base_commit 归属（15 仓库 × 250 任务全命中, `BUILT=250 MISSING=0`）
- `archive_commit`：`git archive --format=tar` → `extractall(filter="data")`（安全解包）
- 每个 bundle：`<task_id>/repo/` + `private/hidden_tests/` 占位 + `task_manifest.json`（runtime 化 task: `image=sweforge-repair:py311`、`workspace=/workspace`、`runtime_user=1000:1000`、`seed_from_snapshot=True`; metadata `bundle_replayed`/`bundle_protocol=exact-mutation-replay-v2`; `integrity_protected` 从 f2p/p2p 测试文件名提取）
- **不触碰 work 仓库状态**（只读 git archive）

产物：`bundles_v1/` 129MB（250 个 task 目录）→ `bundles_v1.tgz` 18MB。

### 5.3 Mac 侧同步（5 步, 用户操作——AutoDL→Mac 无通道）

```bash
# 1. 下载（scp 或 AutoDL 网页文件管理）
scp root@<autodl-ip>:<端口>/root/autodl-tmp/experiments/rl_formal_v1/bundles_v1.tgz ~/
# 2. 解包（新建目录, 不动现有 examples）
cd ~/code/SWE_project && mkdir -p bundles_v1 && tar -xzf ~/bundles_v1.tgz -C bundles_v1
# 3. 确认镜像（manifest 要求 sweforge-repair:py311）
docker images | grep sweforge-repair   # 没有就 tag 现有修复镜像: docker tag <旧名> sweforge-repair:py311
# 4. 重启 server（--bundles-dir 指向新目录; 停掉旧的）
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY NO_PROXY=127.0.0.1,localhost \
PYTHONPATH=src /opt/homebrew/Caskroom/miniconda/base/bin/python -m sweforge.env_server.server \
  --bundles-dir bundles_v1 --docker --cleanup-stale 3600 --port 8501
# 5. 确认隧道（若断重建）
ssh -N -R 8500:127.0.0.1:8501 -i ~/.ssh/autodl_ed25519 -p <端口> root@<autodl>
```

## 六、验证方法（同步后的第一件事）

1. **gold-patch 重验**：`PYTHONPATH=src python scripts/probe_gold_patch_reward.py` → 3 条任务应全部 `resolved, reward=1.0`（`RESULT: PASS`）。这关不过不重训。
2. **重跑阶段② v2**（pool_v2, 5 step）→ 预期出现**非零 reward、advantage 有方差、pg_loss 非零**——此时才进入阶段③ 50-step 正式训练。

## 七、经验沉淀（本项目卡点模式）

1. **reward 恒 0 先验环境有效性, 再谈 reward 设计与模型能力**。判别顺序：空 patch（验判据）→ gold patch（验环境）→ 容器直查（验事实）。gold patch 是免费的正例对照——它"必然正确", 它过不了就是环境问题。
2. **"链路无报错"不等于"环境有效"**。多轮 AgentLoop + 工具 + export_patch 全部健康, 测试却在容器里假失败——架构健康与数据就绪是两回事。检查点要下探到"测试真的跑起来了吗"。
3. **bundle 构建与 bundle 同步是两个动作**。工厂 verified 只在 AutoDL 侧证明任务可解; Mac server 的 `--bundles-dir` 必须另行同步。正式实验前先做 bundles-dir 内容核对（`<task_id>/repo` 存在性抽样）。
4. **AutoDL 侧看不见 Mac 实现**（LocalDocker/bundles 查找在 Mac 的 `sweforge.env_server`, 不在本 repo）→ 诊断只能靠运行时证据, 不要 grep 本 repo 找 bundle 逻辑（找不到是正常的, 不是搜错了）。
5. **工厂 work 仓库是金矿, 别急着清理**：base_commit 全量命中的代码快照直接决定能否低成本修复环境; `git cat-file` 判定归属 + `git archive` 抽取是通用回捞手段。
6. 描述丰富化（`scripts/enrich_pool_descriptions.py`, pool_v2）仍是有效改进（模型行为确实变化）, 但它**只能放大信号, 不能创造信号**——环境没代码时一切都是徒劳。两类问题要分开排障。
7. 问题域扩展提醒：同一陷阱可能出现在 hidden tests 缺失（Mac 侧 `private/hidden_tests/` 注入是隐式步骤）、镜像 tag 漂移（manifest image ≠ 实际镜像）。验证都靠 gold patch 探针兜底。

---

## 附：本卡点关键文件

| 文件 | 角色 |
|---|---|
| `scripts/probe_gold_patch_reward.py` | 判别探针（空 patch vs gold patch, 一次性跑完给出 PASS/FAIL 结论） |
| `scripts/build_grpo_bundles.py` | 修复工具（work repos → bundle, `BUILT=250 MISSING=0`） |
| `scripts/enrich_pool_descriptions.py` | 配套改进（pool_v2 具体化描述, 非根因修复） |
| `scripts/build_ast_bundles.py` | bundle 协议原定义（manifest 字段来源） |
| `src/sweforge/environment/server.py` | §8 契约 server（AutoDL 侧唯一入口; `:155-159` verify 委托） |
| Mac `sweforge.env_server.server` | `--bundles-dir` / LocalDocker 实现（**不在本 repo**） |

---

## 八、第二层根因：数据/生成器层（bundles 修好后探针仍 FAIL 的三个问题）

bundles 部署 + schema 容错后, gold-patch 探针依然 FAIL（P2P 几乎全过但 F2P 恒 False、`integrity=False`）——**环境好了, 数据/生成器坏**。三个问题按探针暴露顺序：

### 8.1 gold_patch 缺尾换行（git apply corrupt）

- 现象: `git apply` 报 `error: corrupt patch at line N`（正向/反向都失败）。
- 根因: `reversal.py:61` `git_cmd` 返回 `proc.stdout.strip()` —— git diff 输出的尾 `\n` 被剥掉。
- 影响: gold_patch 永远无法直接 apply -> `integrity=False`; **模型产出的 patch 同样可能缺尾 \n** -> CleanVerifier `git apply` corrupt -> 正确修复也拿 0（潜在训练杀手）。
- 修复: `verifier.py` `apply_patch` 补规范化（`patch.endswith("\n") or patch += "\n"`）——**任何消费方 apply 前都必须补**; `factory.py` 生成时补回; `fix_pool_quality.py` 数据修复时补。

### 8.2 gold_patch 方向语义不一致（变异 diff vs 修复 diff）

- 现象: 探针把 gold_patch 当"应用即修复"用, apply 失败。
- 根因: 工厂原设计（factory.py:11）mutation 任务 gold_patch = **变异 diff**（`git diff fix_commit base_commit`, 原始->变异）, reversal 任务 = **fix diff**（buggy->fixed）——**两种语义并存, 消费方按统一语义使用必错**。
- 修复: 统一为 **"应用即修复"**（修复 diff）: `factory.py:178` 改为 `git diff base_commit fix_commit`; `fix_pool_quality.py` 对存量数据反转; 探针 docstring 声明语义。
- 教训: 数据字段的语义必须全局唯一; "方向"类字段（diff 正反）是隐性契约, 每个新消费方都可能踩。

### 8.3 构建产物依赖测试污染 P2P/F2P（96+2 条任务不可解）

- 现象: toolz 任务的 `test_has_version` 恒失败——P2P 里 96 条任务永远 unresolved（binary 要求 P2P 全过）, F2P 里 2 条任务永远 unresolved。
- 根因: 测试依赖 setuptools_scm 生成的 `toolz/_version.py`（未跟踪文件, 工厂验证时 `pip install -e .` 生成; **bundle 是 git archive 裸源码快照, 没有它**）-> 版本回退 `'0.0.0'` -> 断言失败。工厂验证环境（安装后）与 bundle 运行环境（裸快照）不一致, 分类器把"安装后环境才成立"的测试划进 F2P/P2P。
- 修复: `fix_pool_quality.py` 通用规则——P2P 含噪音测试 -> 剔除该测试（+ test_commands 同步）; **F2P 含噪音测试 -> 整个任务剔除**（F2P 无法替换, 变异破坏的就是版本测试本身, 属伪任务）。
- 教训: 依赖构建产物的测试（version/scm/git-describe 类）在"裸快照 + 清单驱动"的运行模型下恒失败; 分类器应排除, 或 bundle 构建须包含构建产物。

### 8.4 修复链与最终验证（2026-08-10）

| 环节 | 修复 |
|---|---|
| 运行期防御 | `verifier.py` apply_patch 补尾换行规范化 |
| 数据修复 | `scripts/fix_pool_quality.py` -> `full_v3/pool_v3.jsonl`: 248 条（gold 补\n+反转; 96 条 P2P 剔除 + test_commands 同步; 2 条伪任务剔除; 全量断言 f2p/p2p 非空 + test_commands 全覆盖） |
| 生成器 | `factory.py:178` gold 方向统一 + 补 \n; docstring 同步 |
| 探针 | 语义声明"应用即修复"; 默认指向 v3 |
| 验证 | gold-patch 探针 3/3 `resolved=True, reward=1.0`; 空 patch 3/3 `unresolved, reward=0.0`——环境+数据+reward 链路全通 |

### 8.5 经验补充（本卡点追加）

1. **verifier 对清单外测试判失败是特性不是 bug**（f2p/p2p 条目无 test_commands -> exit 1）——它把数据不一致变成显式失败, 也让"F2P 剔漏"立刻现形（Mac 侧正是靠它发现的）。
2. 排查"奖励恒 0"的检查清单（本卡点全流程）: bundles 存在性 -> schema 兼容 -> gold apply（尾\n/方向）-> 测试可执行（构建产物依赖）-> 清单与命令一致 -> 最后才轮到模型能力。
3. 伪任务（F2P 依赖构建产物）无法通过"换 F2P"修复——替换违背 F2P 划分的零伪造语义; 剔除是唯一正确解。
