"""EnvironmentBackend 接口（PROJECT_SPEC §7）—— Agent Loop 与环境的唯一契约。

    class EnvironmentBackend:
        create(task) -> env
        reset(env) -> None
        execute(env, action: ToolAction) -> ToolObservation
        export_patch(env) -> str            # unified diff
        verify(task, patch) -> VerificationResult   # 默认走 clean verifier
        destroy(env) -> None

铁律（P3）: Agent Loop 只依赖此接口，禁止直接依赖 Docker。

实现:
    阶段 5  MockEnvironmentBackend（本地临时目录, 真实文件语义, 无容器隔离）
    阶段 6  RemoteEnvironmentBackend（HTTP API, AutoDL <-> Mac 契约, 阶段 8 联调）
    未来    LocalDockerBackend（Mac 本地 Docker）

env 句柄对后端私有（Mock = 工作区对象, Remote = env_id），接口不假设其类型。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sweforge.protocol.tools import ToolAction, ToolObservation
from sweforge.schemas.task import TaskSpec
from sweforge.schemas.verification import VerificationResult

__all__ = ["EnvironmentBackend"]


class EnvironmentBackend(ABC):
    """环境后端抽象。create 的输入是 server 侧 TaskSpec（含 environment 规格）。"""

    @abstractmethod
    def create(self, task: TaskSpec) -> Any:
        """按 TaskSpec 创建环境（镜像/工作区 + setup_commands），返回后端私有 env 句柄。"""

    @abstractmethod
    def reset(self, env: Any) -> None:
        """把 env 恢复到 create 后的初始状态（丢弃 agent 的所有修改）。"""

    @abstractmethod
    def execute(self, env: Any, action: ToolAction) -> ToolObservation:
        """执行一条 canonical 工具 action，返回结构化 observation。

        语义由五工具协议定义（protocol/tools.py）: bash 真实执行 /
        search 仓库内文本搜索 / view_file 行号窗口 / str_replace 唯一匹配替换
        （empty old_string = 开头插入或创建文件）/ finish 终止确认。
        """

    @abstractmethod
    def export_patch(self, env: Any) -> str:
        """导出当前工作区相对初始状态（base_commit）的 unified diff。

        格式为标准 git diff（a/ b/ 前缀, @@ hunk），可直接被 git apply 消费。
        """

    def verify(self, task: TaskSpec, patch: str) -> VerificationResult:
        """默认实现走 clean verifier（reward/verifier.py, 阶段 9）——后端
        委托 CleanVerifier 并提供自己的 Isolation（Mock -> TempDirIsolation
        seed_files; Mac -> DockerIsolation, 阶段 12）。

        铁律（P4/P5）: 不伪造 verifier reward —— VerificationResult 必须来自
        真实执行（隔离环境应用 patch + 跑 build/test_commands），不得根据
        patch 内容猜测结论; resolved = F2P/P2P 全过 AND integrity_ok。
        """
        raise NotImplementedError(
            "该后端未提供 verify（实现应委托 CleanVerifier, 见 reward/verifier.py）"
        )

    @abstractmethod
    def destroy(self, env: Any) -> None:
        """销毁 env，释放资源（临时目录 / 远程容器）。幂等。"""
