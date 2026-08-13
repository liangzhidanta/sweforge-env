# Mixed RL Task Pool v2 Implementation Plan

**Goal:** 构建可直接供后续 GRPO 使用的 200 条训练任务池，并严格满足 Public Executable 100 条（50%）、AST Mutation 50 条（25%）、Repair Reversal 50 条（25%）。评测任务与训练任务按仓库隔离，且不计入上述 200 条。

## 交付物

- `dataset_v2/train_tasks.jsonl`：200 条完整 `TaskSpec`。
- `dataset_v2/train_pool.jsonl`：与训练加载器兼容的轻量任务池。
- `dataset_v2/train_manifest.json`：来源、仓库、构造类型、测试数量和校验结果统计。
- Public 任务：具备真实容器镜像、F2P/P2P 测试命令，并通过 base/fix 双态验证。
- Repair Reversal：从真实修复 commit 反转得到，具备可复现 parent/fix 状态和 F2P/P2P 测试。
- AST：从现有已验证池中按仓库与 mutation 类型分层抽取 50 条。
- 更新 `docs/SWE-Forge_GRPO训练数据构造与分布说明.md`。

## 实施顺序

1. 为 mixed-pool 定额选择与校验逻辑补失败测试：精确配额、确定性抽样、来源不足报错、去重和 train/eval 仓库隔离。
2. 实现 mixed-pool builder，并让单元测试通过。
3. 适配 Public Executable 任务的镜像原生工作目录和运行用户；补后端测试，保证 Mac Docker 可执行且不回退 Mock。
4. 从 R2E-Gym-Lite 候选中逐条执行 base/fix 验证，筛得 100 条 Public 任务；任务镜像按需拉取并在验证后精确删除，控制 Mac 磁盘占用。
5. 扫描 humanize、toolz、python-dateutil 的历史修复提交，运行 parent/fix 测试分类，筛得 50 条 Repair Reversal 任务。
6. 从 v1 AST 池中按 repo/mutation_type 分层抽取 50 条。
7. 合并并输出 v2：Public 100 + AST 50 + Reversal 50；运行 schema、配额、重复、泄漏、命令可移植性与抽样 Docker 回归验证。
8. 更新数据说明文档，明确实际数量、分布、路径、生成命令、验证证据，以及数据飞轮当前是否已参与本轮构造。

## 完成标准

- `train_tasks.jsonl` 恰好 200 条，来源比例严格为 50% / 25% / 25%。
- 任务 ID 唯一，训练与评测仓库不重叠，无 gold patch / hidden test 进入 policy prompt。
- Public 与 Reversal 均不是仅做字段转换，而是有可执行 base/fix 证据。
- 项目测试通过，产物清单可复现，文档数字与生成文件一致。

