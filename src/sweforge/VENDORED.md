# Vendored AutoDL canonical contract

以下目录/文件是 **AutoDL 侧 canonical 协议**的逐字节拷贝（verbatim），
来源: AutoDL 服务器 `/root/autodl-tmp/SWE_project/src/sweforge/`，
2026-08-07（AutoDL 阶段 14）.

- `schemas/`          — TaskSpec / AgentTrajectory / VerificationResult（pydantic）
- `protocol/`         — 五工具 action/observation、canonical 消息、序列化、校验
- `environment/`      — base(EnvironmentBackend) + Mock + Remote + §8 Server(make_app)
- `reward/verifier.py`— CleanVerifier + Isolation（TempDirIsolation / DockerIsolation）
- `agent/loop.py`     — AgentLoop（AutoDL <-> Mac 联调用, 不依赖 Docker）
- `__init__.py`       — AutoDL 顶层再导出

规则（防漂移）:
1. **不要编辑这些文件**。协议变更是 AutoDL 侧的事，改动请回 upstream。
2. 唯一例外: `environment/taskenv.py` 未 vendor（依赖 AutoDL 侧 `data.factory`，
   Mac 不需要）。
3. 与 upstream 的漂移检查: `diff -r <autodl>/src/sweforge/<pkg> src/sweforge/<pkg>`。

Mac 侧自己的实现（可自由修改）: `env_server/docker/` —— LocalDockerBackend 实现
`environment/base.EnvironmentBackend` 接口，供 `environment/server.make_app` 使用。
