# SWE-Forge Environment Server — M1 Design

- Date: 2026-08-08
- Status: Approved scope (M1); R1+R2 landed 2026-08-08（见下文 status update）

> **2026-08-08 status update（R1+R2 落地后，本文档部分内容已被取代）**
>
> - **协议**: AutoDL canonical 契约已逐字节引入（`schemas/`、`protocol/`、`environment/`、
>   `reward/`、`agent/`，见 `src/sweforge/VENDORED.md`）。下文 §2 描述的老式
>   `schemas.py` dataclass `ToolAction(tool, arguments, request_id)` **已废弃**，
>   以 vendored pydantic 五工具 discriminated union 为准（含 `finish`）。
> - **Docker**: 本机已通过 colima 装好（VM 方案, daemon 出网走宿主机代理）。
>   下文"`docker` 未安装"与"M2 才接 DockerExecutor 完整流程" **不再成立**；
>   `DockerExecutor`（`--user 1000:1000`、`--network none`、tar+`docker cp` 注入
>   快照）已可用，demo/server 加 `--docker` 即走真实容器。
> - **Server / 联调**: §8 FastAPI server（vendored `environment.server.make_app`）
>   已用 `env_server/server.py`（Mac 侧入口, 包 LocalDockerBackend）接上；
>   `interop_mac.py` 本地 E2E + AgentLoop `validate_trajectory` 全链路通过。
>   这些原列为 M2 范围（"FastAPI /v1 路由、SSH tunnel、AutoDL E2E"）已提前落地。
> - 以上现状以代码与 `USAGE.txt` 为准；本文档保留原始 M1 设计决策供追溯。

## 1. Context

SWE-Forge 分两侧：**AutoDL**（模型 / SFT / Agent Loop / Rollout / GRPO）与 **Mac**（执行环境：Environment Server / LocalDockerBackend / Docker Manager / Task Containers / Clean Verifier / Task Registry）。本设计只覆盖 Mac 侧的 **M1** 里程碑，对应任务书的开发顺序 1/3/4/6/7 及"现在首先完成"部分：

1. 共享 Schema / Canonical Agent Protocol 定义
2. Toy repo（纯 Python、ARM64 兼容、缓存 bug）
3. LocalDockerBackend
4. ToolAction / Observation 集成测试（含 validate_trajectory）

M2+ 原列项中 **已提前落地**（见顶部 status update）：FastAPI /v1 server、request_id 幂等、DockerExecutor 完整接线、SSH tunnel 拓扑、AutoDL E2E（interop_mac.py 本地全链路）。仍属 M2+：Bearer auth、Task Registry（Mac 私有 secrets）、Clean Verifier 服务化、DockerExecutor stale cleanup、跨机 SSH tunnel + 真机 AutoDL host 对接。

**本机约束**：Mac 为 arm64，`docker` 未安装。因此 LocalDockerBackend 采用 executor 策略——默认 `LocalExecutor`（本机可跑），检测到 Docker 时可用 `DockerExecutor`（真实容器）。两种 executor 共享同一套 tool 语义与 Observation 输出，保证 AutoDL 对接代码不因 host 类型变化。

**协议对账**：AutoDL 侧 schema（`PROJECT_SPEC.txt`、`src/sweforge/schemas/`、`protocol/`、`environment/base.py`、`validate_trajectory`）暂不可得。M1 按任务书 §1/§8–§12 的字段定义 canonical 协议，并保留为单点可 diff 模块（`schemas.py`）。拿到 AutoDL 代码后逐字段核对调整，Mac 侧不改动业务逻辑结构。

## 2. Canonical Protocol

单一共享契约，位于 `src/sweforge/schemas.py`。所有模型为 dataclass、可 JSON 序列化。

### ToolName
`bash` / `search` / `view_file` / `str_replace`。`finish` 不由 Mac 实现（由 AutoDL Agent Loop 处理）。

### ToolAction
```python
ToolAction(tool: str, arguments: Mapping[str, Any], request_id: str = "")
```
`request_id` 由客户端生成，供幂等与轨迹校验使用。注意：canonical 协议里 ToolAction 不携带
request_id——幂等由 server 传输层处理（vendored `environment/server.py` 的 idempotency cache，已落地）。

### Observation（§8 字段，结构化、稳定、长度受控）
```python
Observation(
    request_id: str,
    env_id: str,
    tool: str,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    content: str,       # 归一化结果（view 渲染 / search 行 / str_replace 结果）
    truncated: bool,
    duration_ms: int,
)
```

### TaskSpec（Policy-visible，§17，不含 secrets）
`task_id / repo / base_commit / problem_statement / test_command / fail_to_pass / pass_to_pass / protected_paths / platform / image / metadata`。**不包含** hidden_tests、gold_patch、verifier_config、generation_metadata。

### VerificationResult（§15）
`f2p_passed / f2p_total / f2p_ratio / p2p_passed / p2p_total / p2p_ratio / integrity_ok / resolved / reward / timeout / details`。
基线 `resolved = 全 F2P 通过 AND 全 P2P 通过 AND integrity_ok`。

### 轨迹
`TrajectoryStep(action: ToolAction, observation: Observation)`；`AgentTrajectory(task_id, steps, metadata)`。

`src/sweforge/protocol.py` 提供 `validate_trajectory()`（返回错误列表，空 = 合法）与 JSON 序列化/反序列化。校验点：tool 名合法、参数符合各工具 schema、Observation 字段完整、同 env 内 request_id 不重且有序、时长/截断标记类型正确。

## 3. Backend 架构

```
src/sweforge/
├── schemas.py          # canonical protocol
├── protocol.py         # validate_trajectory, serialize
└── env_server/
    └── docker/
        ├── backend.py  # LocalDockerBackend + EnvironmentBackend 协议
        └── manager.py  # labels / limits + stale-container cleanup（list/cleanup，server --cleanup-stale）
```

`EnvironmentBackend` 协议（§4）：
```python
create(task: TaskSpec) -> EnvHandle
reset(env: EnvHandle) -> str          # 返回 problem_statement（prompt），非工具调用
execute(env: EnvHandle, action: ToolAction) -> Observation
export_patch(env: EnvHandle) -> str
verify(task: TaskSpec, patch: str) -> VerificationResult
destroy(env: EnvHandle) -> None
```
`reset` 返回问题描述字符串（对应 TRL `reset` 返回 prompt 的既有模式），不产 Observation。

`EnvHandle`：`env_id` + 内部 workspace 状态。M1 状态机只维护创建/就绪/执行中/销毁（完整 CREATED→READY→RUNNING→FINISHED→FAILED→DESTROYED 由 M2 服务层接管）。

### Executor 策略
- `LocalExecutor`：对 workspace 目录直接执行（subprocess / ripgrep 或 fallback 正则 / 路径策略）。用于本机测试与 Mac 开发。M1 的 create/execute/verify 全部走它。
- `DockerExecutor`（M1 提供类 + create-command 构造单测，按 `docker` 可用性门控）：每 env 一个容器，`--network none`、非 root、CPU/内存/PID 限制、`sweforge.managed=true` 等 labels（§19–§20）。`create()` 接入 DockerExecutor 的完整流程（镜像构建、repo 注入、stale cleanup）留到 M2。
- 两者共用 `PathPolicy`（防 `../`、symlink escape、访问私有 bundle）与工具实现，仅"命令如何跑"不同。

## 4. Tools（§9–§12）

- **bash**：`command` + `timeout`。命令经 `/bin/sh -c` 执行（支持管道/重定向），限制 runtime / stdout / stderr（默认 20k 字符截断，置 `truncated`）。安全边界：Docker executor 依赖容器隔离（非 root、无网络、资源限制，§19）；LocalExecutor 仅用于开发/测试，同样强制超时与输出截断。
- **search**：限定 workspace 内、`max_results`（默认 50）与输出长度受控，返回 `path:line:text`（与 ripgrep 输出格式一致）。M1 用 stdlib 正则扫描保证本机确定性；换用 ripgrep 只改实现不动契约，M2 在 Docker 镜像内启用 rg。
- **view_file**：`path`/`start_line`/`end_line`，行号从 1 起，最大 200 行。路径经 PathPolicy 解析，拒绝越界与保护路径。
- **str_replace**：`path`/`old`/`new`/`expected_occurrences`。0 匹配 / 多匹配 → 明确 Observation（error + content），不猜测修改位置。

所有工具失败/非法都返回结构化 Observation（不抛异常到外层）。

## 5. Clean Verifier（M1 本地回退）

语义与 §15 一致，只是用"全新 workspace 副本"替代"fresh container"：

1. 从 task 的 canonical buggy snapshot 复制出全新 workspace
2. `export_patch` 得到的 git diff → 应用到新 workspace
3. 注入 hidden tests（F2P）
4. 跑 F2P → P2P → integrity（受保护路径哈希比对、补丁大小/路径越界检查）
5. 产出 `VerificationResult`

M1 的 integrity 实现最小集：protected path 哈希比对 + patch 内路径越界检查（§16 完整清单留给 M2 verifier）。

## 6. Toy Repo

`examples/toy_cache/`，纯 Python 无第三方依赖：

- `repo/toy_cache/cache.py`：`get_or_compute(key, compute)` 按 key 缓存。**bug**：把值存在 `compute()` 的结果（`value`）名下而不是 `key` 名下 → 两个不同 key 计算出相同值时会互相串值。
- `repo/tests/test_public.py`（P2P，agent 可见）：单 key 重复调用返回相同值；返回正确值。buggy 与 fixed 均通过。
- `private/hidden_tests/test_f2p.py`（Mac 私有）：两个不同 key 计算出相同值，断言二者分别触发 compute（buggy 下第二次命中错误缓存不触发 → fail；fixed 通过）。

验证矩阵：
- buggy：F2P fail，P2P pass
- fixed（应用修复 diff）：F2P pass，P2P pass

## 7. Testing

- `pytest`，`tests/` 下：schema/protocol 单测、PathPolicy 单测、各 tool 单测（含越界/多匹配/截断）、verifier 单测。
- 核心集成测试 `test_rollout_loop.py`：创建 env → `ToolAction → Observation` 循环 → `export_patch` → `verify` → `VerificationResult` → `validate_trajectory()` 通过。此即 §23 集成标准的本地形态。
- Docker executor 测试用 `pytest.mark.skipif(no docker)`。

## 8. Out of Scope (M2+)

已落地（见顶部 status update 与 `USAGE.txt`）：FastAPI `/v1` 路由（vendored `make_app` + `env_server/server.py`）、request_id 幂等、DockerExecutor 完整接线、SSH tunnel 拓扑、AutoDL E2E（interop_mac.py 本地全链路通过）、DockerExecutor stale cleanup（`manager.cleanup_stale_containers` + server `--cleanup-stale SECONDS`）、Bearer auth（server `--token`/`SWEFORGE_TOKEN` + `env_server/client.py` 参考实现，AutoDL 客户端加一行请求头即接入）、per-task 镜像解析（`task.environment.image` 优先于默认镜像）。

仍属 M2+：Task Registry（Mac 私有 secrets，当前 bundles_dir 即简化注册表）、Clean Verifier 服务化（已作为 `/v1/verifications` 端点）、SSH reverse tunnel 真机建立、AutoDL 跨机 host 真机对接（`validate_trajectory()` 已在本地 E2E 验证）。
