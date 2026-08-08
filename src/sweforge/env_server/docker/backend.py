"""LocalDockerBackend — AutoDL EnvironmentBackend 的 Mac 侧实现（阶段 8 联调用）。

create/reset/execute/export_patch/verify/destroy 语义与 vendored
environment/mock.py 一致（五工具 canonical 观察渲染、finish 带 patch），
区别仅在: 工作区通过 Executor 抽象隔离 —— LocalExecutor（本地临时目录, 无
容器）或 DockerExecutor（每 env 一个非 root、无网络容器, image + seed）。

Bundle 注册表: bundles_dir/<task_id>/ 下 task_manifest.json（canonical
TaskSpec）+ repo/（初始仓库快照）+ private/hidden_tests/（F2P 隐藏测试,
policy 不可见）。verify 时以 bundle 里的 task 为权威来源（P5: 调用方伪造的
TaskSpec 不能削弱完整性）。
"""

from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from sweforge.environment.base import EnvironmentBackend
from sweforge.env_server.docker.executors import DockerExecutor, Executor, LocalExecutor
from sweforge.env_server.docker.tools import execute_action
from sweforge.env_server.docker.verify import verify_clean
from sweforge.protocol.tools import FinishAction, ToolAction, ToolObservation
from sweforge.schemas.task import TaskSpec
from sweforge.schemas.verification import VerificationResult

__all__ = [
    "TaskBundle",
    "EnvHandle",
    "LocalDockerBackend",
    "load_task_bundle",
    "git_init_workspace",
]

_DEFAULT_SETUP_TIMEOUT = 120.0


@dataclass
class TaskBundle:
    task: TaskSpec
    repo_path: Path
    hidden_tests: Path
    integrity_protected: tuple[str, ...] = ()


@dataclass
class EnvHandle:
    env_id: str
    task: TaskSpec
    executor: Executor
    finished: bool = False
    closed: bool = False


def load_task_bundle(bundle_dir: Path) -> TaskBundle:
    manifest = json.loads((bundle_dir / "task_manifest.json").read_text(encoding="utf-8"))
    task = TaskSpec.model_validate(manifest["task"])
    repo_path = bundle_dir / "repo"
    hidden_tests = bundle_dir / "private" / "hidden_tests"
    if not repo_path.is_dir():
        raise FileNotFoundError(f"bundle {bundle_dir} missing repo/")
    return TaskBundle(
        task=task, repo_path=repo_path, hidden_tests=hidden_tests,
        integrity_protected=tuple(manifest.get("integrity_protected", [])),
    )


def git_init_workspace(executor: Executor) -> None:
    """Commit a baseline after setup so export_patch never emits setup artifacts."""
    executor.write_text(".gitignore", "__pycache__/\n*.pyc\n.pytest_cache/\n")
    executor.run_argv(("git", "init", "-q"), timeout=30)
    executor.run_argv(("git", "config", "user.email", "sweforge@localhost"), timeout=10)
    executor.run_argv(("git", "config", "user.name", "SWE-Forge"), timeout=10)
    executor.run_argv(("git", "add", "-A"), timeout=60)
    executor.run_argv(("git", "commit", "-q", "-m", "baseline"), timeout=60)


class LocalDockerBackend(EnvironmentBackend):
    """AutoDL EnvironmentBackend on Mac: LocalExecutor (dev) or DockerExecutor.

    bundles_dir: task bundle registry (see module docstring). use_docker=True
    requires a running Docker daemon and the base image (see Dockerfile).
    """

    def __init__(
        self,
        bundles_dir: str | Path,
        use_docker: bool = False,
        image: str = "sweforge-base",
        docker_binary: str = "docker",
        max_output_chars: int = 20_000,
        max_view_lines: int = 200,
        setup_timeout: float = _DEFAULT_SETUP_TIMEOUT,
    ) -> None:
        self.bundles_dir = Path(bundles_dir)
        self.use_docker = use_docker
        self.image = image
        self.docker_binary = docker_binary
        self.max_output_chars = max_output_chars
        self.max_view_lines = max_view_lines
        self.setup_timeout = setup_timeout
        # 无 bundle 注册的任务: 空快照 + setup_commands 自建环境（AutoDL Mock 语义）
        self._empty_snapshot = Path(tempfile.mkdtemp(prefix="sweforge-empty-snapshot-"))

    # ------------------------- executor 构造 -------------------------

    def _make_executor(self, bundle: TaskBundle, role: str) -> Executor:
        """Fresh executor whose root holds the bundle repo snapshot (no hidden tests)."""
        if self.use_docker:
            docker = DockerExecutor(
                image=self.image,
                container_name=f"sweforge-{role}-{bundle.task.task_id[:12]}-{uuid.uuid4().hex[:8]}",
                task_id=bundle.task.task_id,
                env_id=bundle.task.task_id,
                docker_binary=self.docker_binary,
                max_output_chars=self.max_output_chars,
                max_view_lines=self.max_view_lines,
            )
            docker.seed_from_snapshot(bundle.repo_path)
            return docker
        return LocalExecutor.from_snapshot(
            bundle.repo_path,
            max_output_chars=self.max_output_chars,
            max_view_lines=self.max_view_lines,
        )

    def _bundle(self, task: TaskSpec) -> TaskBundle:
        """Bundle registry lookup, falling back to AutoDL Mock semantics.

        Registered tasks return the authoritative bundle (task + repo snapshot +
        hidden tests). Unregistered tasks run setup-driven: an empty snapshot
        workspace that setup_commands populate, no hidden tests — matching how
        AutoDL's mock backend creates an environment from a bare TaskSpec.
        """
        bundle_dir = self.bundles_dir / task.task_id
        if (bundle_dir / "task_manifest.json").is_file():
            return load_task_bundle(bundle_dir)
        return TaskBundle(
            task=task,
            repo_path=self._empty_snapshot,
            hidden_tests=self._empty_snapshot / ".no-hidden-tests",
            integrity_protected=(),
        )

    def _protected_paths(self, bundle: TaskBundle) -> tuple[str, ...]:
        hidden = (
            tuple(p.relative_to(bundle.hidden_tests).as_posix() for p in bundle.hidden_tests.rglob("*.py"))
            if bundle.hidden_tests.is_dir()
            else ()
        )
        return (".git", *hidden, *bundle.integrity_protected)

    # ------------------------- EnvironmentBackend -------------------------

    def create(self, task: TaskSpec) -> EnvHandle:
        bundle = self._bundle(task)
        executor = self._make_executor(bundle, "task")
        for cmd in bundle.task.environment.setup_commands:
            executor.run_shell(cmd, timeout=self.setup_timeout)
        git_init_workspace(executor)
        return EnvHandle(env_id=task.task_id, task=bundle.task, executor=executor)

    def reset(self, env: EnvHandle) -> None:
        """丢弃 agent 所有修改, 恢复到 create 后初始状态（setup 后基线）。"""
        if env.closed:
            raise RuntimeError(f"env {env.env_id} already destroyed")
        env.executor.close()
        bundle = self._bundle(env.task)
        env.executor = self._make_executor(bundle, "task")
        for cmd in bundle.task.environment.setup_commands:
            env.executor.run_shell(cmd, timeout=self.setup_timeout)
        git_init_workspace(env.executor)
        env.finished = False

    def execute(self, env: EnvHandle, action: ToolAction) -> ToolObservation:
        if env.closed:
            raise RuntimeError(f"env {env.env_id} already destroyed")
        observation = execute_action(env.executor, action, export_patch=lambda: self.export_patch(env))
        if isinstance(action, FinishAction):
            env.finished = True
        return observation

    def export_patch(self, env: EnvHandle) -> str:
        """当前工作区相对 setup 后基线的 unified diff（git apply 可消费）。"""
        if env.closed:
            raise RuntimeError(f"env {env.env_id} already destroyed")
        env.executor.run_argv(("git", "add", "-N", "."), timeout=30)
        result = env.executor.run_argv(("git", "diff", "--no-ext-diff", "--", "."), timeout=30)
        if result.timed_out:
            raise TimeoutError("git diff timed out")
        return result.stdout

    def verify(self, task: TaskSpec, patch: str) -> VerificationResult:
        """clean-container verify（P4）: bundle 为权威来源, 隔离环境应用 patch。"""
        bundle = self._bundle(task)
        protected = self._protected_paths(bundle)
        return verify_clean(
            bundle.task,
            patch,
            make_executor=lambda: self._make_executor(bundle, "verify"),
            hidden_tests=bundle.hidden_tests if bundle.hidden_tests.is_dir() else None,
            protected_paths=protected,
            metadata_extra={"backend": "local-docker", "docker": self.use_docker},
        )

    def destroy(self, env: EnvHandle) -> None:
        if env.closed:
            return
        env.executor.close()
        env.closed = True
