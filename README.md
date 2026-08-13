# sweforge-env

SWE-Forge 的 **Mac 侧执行环境**：实现 §8 契约的执行层 —— canonical 五工具、git patch 导出、clean verifier（注入隐藏测试），以及每 env 一个隔离 Docker 容器的真实运行后端。

> 训练侧框架（数据工厂 / SFT / GRPO）见 [SWE-Forge](https://github.com/liangzhidanta/sweforge)。

## 在两端拓扑中的位置

```
AutoDL  Agent / CleanVerifier
   └─> RemoteEnvironmentBackend        # 只依赖 EnvironmentBackend 接口
   └─> HTTP API (SSH 隧道; Docker daemon 绝不暴露公网)
   └─> 本包 Environment Server (仅监听 127.0.0.1)
   └─> DockerExecutor → 每 env 一个隔离容器  |  LocalExecutor → 本地临时目录
```

AutoDL 侧只依赖 `EnvironmentBackend` 接口（`create / reset / execute / export_patch / verify / destroy`）；本包实现该接口的真实后端 `LocalDockerBackend`，切换后端不改 Agent / Trainer。

## 特性

- **canonical 五工具**：`bash` / `search` / `view_file` / `str_replace` / `finish`，观察渲染与 AutoDL `mock.py` 逐条镜像（SFT 与 RL 一致）。
- **clean verifier**：全新工作区注入隐藏测试，判定 `resolved = 全部 F2P pass ∧ 全部 P2P pass ∧ integrity_ok`，绝不信任 Agent 工作环境。
- **双后端**：`LocalExecutor`（无需 Docker）与 `DockerExecutor`（每 env 一个 `--network none` 非 root 容器，CPU/内存/PID 限制，带 `sweforge.managed` 标签）。
- **路径安全**：拒绝越界 / symlink / 保护路径；工作区内绝对路径自动归一，工作区外绝对路径拒绝。
- **每任务镜像**：`TaskSpec.environment.image` 指定预装依赖镜像（容器无网，依赖必须预先打进镜像；不指定则用默认 `sweforge-base`，仅 git + pytest）。
- **认证与幂等**：可选 Bearer token（`--token` / `SWEFORGE_TOKEN`）；所有修改状态请求携带 `request_id`，server 幂等表去重。
- **泄漏清理**：`--cleanup-stale SEC` 启动时按容器年龄清理崩溃残留的 managed 容器。
- **vendored 契约**：`schemas/ protocol/ environment/ reward/ agent/` 为 AutoDL canonical 逐字节拷贝，与训练侧零漂移（见 `src/sweforge/VENDORED.md`）。

## 快速开始

```bash
# 环境要求: macOS (arm64), Python >= 3.11 (本机 base conda 3.12), 可选 Docker/colima

# 1) 构建默认基础镜像 (仅 Docker 后端需要)
docker build -t sweforge-base src/sweforge/env_server/docker/

# 2) 起 server (开发模式, LocalExecutor, 无需 Docker)
NO_PROXY=127.0.0.1,localhost PYTHONPATH=src \
  python -m sweforge.env_server.server --bundles-dir examples

# 3) 端到端演示: 建 env -> 看源码 -> 定位 bug -> 修复 -> 导出 patch -> 干净验证
NO_PROXY=127.0.0.1,localhost PYTHONPATH=src python examples/demo_rollout.py
#   Docker 版加 --docker

# 4) 测试与风格
NO_PROXY=127.0.0.1,localhost python -m pytest -q     # 全部测试
python -m ruff check src tests examples
```

> 本机宿主 Clash 代理会拦 127.0.0.1 → 服务器 / 客户端 / 测试都要加 `NO_PROXY=127.0.0.1,localhost`。

## Server 参数

| 参数 | 说明 |
|---|---|
| `--bundles-dir DIR` | task bundle 注册表根目录（`<task_id>/task_manifest.json` + `repo/` + `private/hidden_tests/`）；未登记的 task 走 setup-command 自建工作区 |
| `--docker` | 用 DockerExecutor（每 env 一个非 root、`--network none` 容器）；默认 LocalExecutor |
| `--image IMG` | 默认镜像回退值（task 未指定 `environment.image` 时用），默认 `sweforge-base` |
| `--port PORT` | 监听端口，默认 8500 |
| `--token SECRET` | 可选 Bearer 认证；缺省读环境变量 `SWEFORGE_TOKEN` |
| `--cleanup-stale SEC` | 启动前清理早于 SEC 秒创建的 sweforge-managed 泄漏容器 |
| `--host HOST` | 绑定地址，默认 `127.0.0.1`（公网访问一律走 SSH tunnel） |

## §8 契约端点

```
GET    /health                     # 存活检查 (不加认证也通)
POST   /v1/tasks/register          # 登记 TaskSpec (server 持有)
POST   /v1/envs                    # create: 内联 {"task"} 或引用 {"task_id"}
POST   /v1/envs/{env_id}/reset
POST   /v1/envs/{env_id}/actions   # execute(action) -> observation
GET    /v1/envs/{env_id}/patch     # export_patch
DELETE /v1/envs/{env_id}           # request_id 放 query
POST   /v1/verifications           # clean-container verify(task, patch)
```

## 目录结构

```
src/sweforge/
  schemas/ protocol/ environment/ reward/ agent/   # vendored AutoDL canonical 契约 (勿改)
  env_server/
    server.py          # Environment Server CLI (vendored make_app + auth + LocalDockerBackend)
    client.py          # AuthenticatedRemoteEnvironmentBackend (参考实现)
    docker/
      path_policy.py   # 路径安全: 拒绝越界/symlink/保护路径
      manager.py       # 容器 labels/limits + stale-container 清理
      executors.py     # Executor 抽象 + LocalExecutor + DockerExecutor
      tools.py         # canonical 五工具观察渲染
      verify.py        # clean verifier (全新工作区注入隐藏测试)
      backend.py       # LocalDockerBackend (含 task.environment.image 解析)
      Dockerfile       # sweforge-base 基础镜像
examples/toy_cache_aliasing/  # 示例任务 bundle (repo/ + private/hidden_tests/ + task_manifest.json)
tests/                # vendored AutoDL 契约测试 + Mac 自有测试
docs/                 # GRPO 卡点沉淀 / 操作手册 / 训练数据可视化
```

## 文档

- `USAGE.txt` — 完整使用说明（含两端联调教程、Docker/colima 配置、常见问题）
- `src/sweforge/VENDORED.md` — vendored 契约边界与防漂移规则
- `docs/` — 卡点沉淀、正式运行手册、GRPO 训练数据说明与可视化图
