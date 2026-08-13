# SWE-Forge formal 任务 Docker 空工作区问题复盘

> 复盘日期：2026-08-10  
> 问题范围：formal RL task pool、Mac Docker Environment Server、CleanVerifier  
> 结论：Docker/Verifier 链路本身有效，但 formal 任务对应的代码 bundle 未部署到 Mac，导致容器得到空 Git 工作区，训练奖励因此恒为 0。

## 1. 问题现象

在 formal RL 任务上运行 Agent rollout 时出现：

- reward 长期为 `0`；
- verifier 返回 `unresolved`；
- F2P/P2P 测试未通过；
- 即使应用 gold patch，`integrity=False`，任务仍无法解决；
- 丰富问题描述后，模型的工具调用行为发生变化，但 reward 仍然为 `0`。

最初可能怀疑：

1. 模型不会调用工具或不会修复代码；
2. reward 函数设计错误；
3. CleanVerifier 判定错误；
4. Docker 中的正式任务环境没有正确构建。

最终检查确认是第 4 项。

## 2. 最终根因

formal pool 的 250 条任务是在 AutoDL 侧构造和验证的，但 Mac Environment Server 当时使用：

```bash
--bundles-dir examples
```

`examples` 中只有 demo bundle，没有 formal pool 中 250 个 `task_id` 对应的代码快照。

正式任务到达 Mac 后，LocalDocker 后端无法找到：

```text
<bundles-dir>/<task_id>/repo/
```

于是创建出来的 `/workspace` 实际只有：

```text
.git/
.gitignore
一个 baseline commit
```

目标仓库的源码、`tests/`、`src/` 和 patch 目标文件均不存在。因此：

- pytest 报 `file or directory not found`，并没有真正运行测试；
- 模型即使给出合理修改，也没有实际代码可以编辑；
- gold patch 找不到目标文件，无法应用；
- CleanVerifier 根据实际执行结果返回 `unresolved` 和 `reward=0`。

数据流如下：

```text
formal pool 中的 task_id
        ↓
Mac Server 在 examples/<task_id>/repo 查找 bundle
        ↓
没有找到 formal task bundle
        ↓
LocalDocker 创建空 Git 工作区
        ↓
源码、测试及 patch 目标文件不存在
        ↓
测试 ERROR / gold patch 应用失败
        ↓
CleanVerifier 正确返回 unresolved
        ↓
reward 恒为 0
```

## 3. 为什么之前的测试能够通过

此前通过的测试验证了链路，但没有覆盖 formal bundle 的完整性。

### 3.1 interop 7/7 验证的内容

此前联通测试成功验证了：

- AutoDL 能通过 SSH 反向隧道访问 Mac Server；
- Mac 能创建真实 Docker 容器；
- Agent 可以执行 bash、查看文件、修改文件和导出 patch；
- CleanVerifier 能执行测试并返回结果；
- resolved 状态和 reward 能正确回传给 AgentLoop/GRPO。

这些结论仍然有效。

### 3.2 测试使用的是 demo/toy task

当时 Environment Server 指向 `examples`，interop 使用的也是其中已有的 demo bundle。另一个 GRPO toy pool 会通过 `setup_commands` 在容器中主动创建 `answer.txt` 和对应验证条件，因此不依赖 formal 仓库快照。

所以 demo/toy task 能形成完整闭环：

```text
已有 demo bundle，或 setup_commands 创建文件
        ↓
Agent 修改真实存在的文件
        ↓
测试真实执行
        ↓
CleanVerifier 返回 resolved=1.0
```

该测试证明的是“执行链路可用”，不是“formal 250 条任务均已部署”。

### 3.3 当时缺失的验收项

测试中没有从 formal pool 随机抽取真实 `task_id`，逐项验证：

- `<bundles-dir>/<task_id>/repo` 是否存在；
- 容器 `/workspace` 是否包含目标仓库源码和测试；
- 当前 HEAD 是否与该任务的 `base_commit` 一致；
- F2P 在 buggy 状态下是否真实失败；
- gold patch 是否能成功应用；
- 应用 gold patch 后 F2P 是否转为通过，同时 P2P 保持通过。

因此出现了“demo 链路通过，但 formal 训练环境无效”的假安全感。

## 4. reward 函数是否有问题

没有证据表明 reward 函数本身有错误。

在空工作区中：

- 测试无法运行；
- patch 无法应用；
- 任务没有被解决。

CleanVerifier 返回 `unresolved`、`reward=0` 在语义上是正确的。问题在于 verifier 验证的是一个错误构建的环境，导致这个 reward 无法衡量模型的真实代码修复能力。

因此应区分：

```text
reward 计算逻辑：正确
reward 所依赖的 formal 执行环境：错误
最终训练信号：数值正确，但语义无效
```

## 5. 影响范围

受影响的是所有在缺失 formal bundle 条件下得到的 formal rollout 结果，包括：

- v1/v2 formal 任务的全零 reward；
- 基于这些 reward 得出的“模型没有修对”判断；
- resolve rate、F2P/P2P 以及 advantage 方差统计；
- 如果曾基于这些轨迹更新策略，相应训练 step 不应作为正式实验结果。

不受影响的部分包括：

- demo/toy 环境的 Docker 联通结论；
- 工具调用协议和 patch 导出链路；
- CleanVerifier 的基本执行与 reward 回传能力；
- 4B GRPO 在 FSDP → vLLM 权重同步阶段的 OOM 诊断。该 OOM 发生在正式 rollout 之前，是另一项独立问题。

## 6. 修复方案

从 AutoDL 数据工厂保留的 work 仓库中，根据每条任务的 `base_commit` 生成与 pool 一一对应的代码快照：

```text
<task_id>/
├── repo/                  # 完整源码和测试，定位到任务 base_commit
└── task_manifest.json     # image、workspace、seed/integrity 配置
```

当前生成结果：

- formal bundle：250/250；
- 压缩包：`/root/autodl-tmp/experiments/rl_formal_v1/bundles_v1.tgz`；
- 生成脚本：`scripts/build_grpo_bundles.py`；
- Mac 解压目标：`/Users/apple/code/SWE_project/bundles_v1`；
- Environment Server 应改为 `--bundles-dir bundles_v1`；
- bundle manifest 使用镜像 `sweforge-repair:py311`。

## 7. 修复后的强制验收顺序

不能直接重启 50-step GRPO。必须依次完成：

### 7.1 静态完整性检查

从 train pool 随机抽取不少于 3 条任务，确认：

- task_id 在 `bundles_v1` 中存在；
- `repo/` 不是空仓库；
- 问题描述涉及的目标文件存在；
- F2P/P2P 指向的测试文件和测试节点存在；
- bundle HEAD/base_commit 与 TaskSpec 一致。

### 7.2 gold-patch 探针

对抽取任务分别执行：

1. 创建 Docker 环境；
2. 验证 buggy 状态下 F2P 失败；
3. 应用 gold patch；
4. 验证 F2P 转为通过；
5. 验证 P2P 仍通过；
6. 要求 `integrity=True`、`resolved=True`、`reward=1.0`。

若 gold patch 探针失败，应继续修环境，不得归因于模型。

### 7.3 空 patch 负向探针

在同一任务上不做修改直接 finish，应该得到：

```text
resolved=False
reward=0.0
```

正负探针共同通过，才能证明环境和 reward 的组合语义正确。

### 7.4 5-step GRPO 稳定性测试

通过环境验收后再运行丰富化描述的 5-step 测试，检查：

- 模型产生真实工具调用；
- 测试实际执行，而不是 file-not-found；
- 同组 rollout 出现 reward 差异；
- advantage 不恒为 0；
- `grad_norm`、`pg_loss` 为有限值；
- 容器、Ray 和 vLLM 能正常清理。

只有以上全部通过，才能进入 50-step 正式训练。

## 8. 后续永久增加的防回归检查

以后每个 task pool 版本在训练前必须做数据—环境一致性审计：

1. `pool task_id` 与 `bundle task_id` 集合完全一致；
2. 每个 bundle 都包含非空 Git 仓库、源码和测试；
3. 每个 `base_commit` 可解析且与 bundle HEAD 对齐；
4. F2P/P2P 测试节点真实存在；
5. 随机 gold-patch 探针全部得到 `reward=1`；
6. 随机空-patch 探针全部得到 `reward=0`；
7. Environment Server 启动日志明确记录所用 bundles 目录和已加载任务数；
8. 正式训练日志记录 task pool 版本、bundle 版本及其 hash。

## 9. 复盘结论

本次问题不是 Docker 引擎不可用，也不是 reward 公式错误，而是 formal task pool 与 Mac 侧 bundle 部署脱节：训练任务存在，但任务对应的代码仓库快照没有进入 Docker。

此前测试之所以通过，是因为它使用了 `examples` 中真实存在的 demo bundle，或由 `setup_commands` 自建文件的 toy task。该测试覆盖了通信、Docker、工具和 verifier 链路，却没有覆盖 formal task 与 bundle 的一一对应关系。

最重要的改进是：今后不能只做“链路测试”，还必须在正式训练前完成“formal task 数据—代码快照—测试—gold patch”的端到端语义验收。
