"""阶段 9: Clean Verifier —— 隔离环境 + git apply + build + test（PROJECT_SPEC §5/P4/P5）。

职责: 在全新隔离环境（绝不信任 Agent 工作区, P4）应用 patch、跑
build/test_commands，产出 VerificationResult。resolved = 全部 F2P pass
AND 全部 P2P pass AND integrity_ok（patch 干净应用, §5）; baseline reward
= binary_reward(v) = 1.0/0.0。

Isolation 抽象:
    TempDirIsolation    toy/CI（AutoDL 侧）: 全新临时目录 + seed_files +
                        setup_commands 的真实文件语义, 无容器隔离
    DockerIsolation     Mac 正式版（阶段 12 实现）: clean container;
                        HTTP 契约不变（POST /v1/verifications 的
                        backend.verify 委托本 verifier）

不伪造 reward（P5）: 每个 TestResult 来自真实命令 exit code; patch 应用
失败 = integrity_ok=False + unresolved（不是猜测结论, 也不是让调用方
崩掉的异常）。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from sweforge.schemas.task import TaskSpec
from sweforge.schemas.verification import (
    TestResult,
    VerificationResult,
    binary_reward,
)

__all__ = [
    "CleanWorkspace",
    "Isolation",
    "TempDirIsolation",
    "DockerIsolation",
    "CleanVerifier",
    "VERIFIER_VERSION",
]

#: verifier 版本（metadata 记录, 溯源用）
VERIFIER_VERSION = "clean-v1"
#: bash 默认超时（秒）
_DEFAULT_BASH_TIMEOUT = 60.0
#: TestResult.output 截断上限（调试用尾部输出）
_OUTPUT_CAP = 2000


# ------------------------- 隔离抽象 -------------------------


@dataclass
class CleanWorkspace:
    """隔离执行环境。path = 可执行工作目录; 容器句柄由 isolation 私有持有。"""

    path: Path
    setup_failures: list[str] = field(default_factory=list)


class Isolation(Protocol):
    """干净执行环境。create 抛异常 -> verify 返回 verdict="error"（verifier 侧故障）。"""

    def create(self, task: TaskSpec) -> CleanWorkspace: ...

    def apply_patch(self, workdir: Path, patch: str) -> str | None:
        """在干净基线上应用 unified diff。None = 干净应用; str = 错误信息
        （-> integrity_ok=False, 不抛异常）。"""

    def destroy(self, workdir: Path) -> None:
        """销毁执行环境（幂等）。"""


class TempDirIsolation:
    """toy/CI 隔离: 全新临时目录 + seed_files + setup_commands（真实文件语义）。

    seed_files 由后端持有（Mock 的种子快照）; Docker 版由 base image +
    base_commit 提供等价基线（阶段 12）。
    """

    def __init__(
        self,
        seed_files: dict[str, str] | None = None,
        bash_timeout: float = _DEFAULT_BASH_TIMEOUT,
    ):
        self.seed_files = dict(seed_files or {})
        self.bash_timeout = bash_timeout

    def create(self, task: TaskSpec) -> CleanWorkspace:
        workdir = Path(tempfile.mkdtemp(prefix="sweforge-clean-"))
        for rel, content in self.seed_files.items():
            p = workdir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        failures: list[str] = []
        env = _child_env(workdir)
        for cmd in task.environment.setup_commands:
            obs = _run_shell(workdir, cmd, self.bash_timeout, env=env)
            if obs.exit_code != 0:
                failures.append(f"{cmd!r} -> exit {obs.exit_code}: {(obs.stderr or obs.stdout)[-200:]}")
        return CleanWorkspace(path=workdir, setup_failures=failures)

    def apply_patch(self, workdir: Path, patch: str) -> str | None:
        if not patch.strip():
            return None  # 空 patch 是合法的"没修", unresolved 由 tests 给出, 非完整性错误
        if subprocess.run(["git", "init", "-q"], cwd=workdir, capture_output=True).returncode != 0:
            return "git init failed"
        proc = subprocess.run(
            ["git", "apply", "-p1", "--whitespace=nowarn", "-"],
            cwd=workdir, input=patch, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return proc.stderr.strip() or "git apply failed"
        return None

    def destroy(self, workdir: Path) -> None:
        shutil.rmtree(workdir, ignore_errors=True)


class DockerIsolation:
    """Mac 侧 clean-container 隔离（P4）—— 阶段 12 实现, AutoDL 侧不可用。

    结构保持: create -> CleanWorkspace(path=容器内挂载点, 容器句柄私有),
    apply_patch -> 容器工作目录内 git apply, destroy -> 移除容器。
    """

    def __init__(self, image: str = "sweforge-base"):
        self.image = image

    def create(self, task: TaskSpec) -> CleanWorkspace:
        raise NotImplementedError(
            "DockerIsolation 由 Mac 侧实现（阶段 12）; AutoDL 侧用 TempDirIsolation"
        )

    def apply_patch(self, workdir: Path, patch: str) -> str | None:
        raise NotImplementedError

    def destroy(self, workdir: Path) -> None:
        raise NotImplementedError


# ------------------------- CleanVerifier -------------------------


class CleanVerifier:
    """clean verifier（§5）: 隔离环境 + git apply + build + test, 不伪造 reward（P5）。"""

    def __init__(self, isolation: Isolation, *, bash_timeout: float = _DEFAULT_BASH_TIMEOUT):
        self.isolation = isolation
        self.bash_timeout = bash_timeout

    def verify(
        self, task: TaskSpec, patch: str, *, metadata_extra: dict | None = None
    ) -> VerificationResult:
        """验证 patch 是否使任务 resolved（真实执行, 不猜测）。

        语义:
            - isolation.create 故障          -> verdict="error"（verifier 侧问题）
            - patch 应用失败                 -> integrity_ok=False + unresolved
            - setup 失败                    -> metadata["setup_failures"]（继续跑,
                                              任务环境问题, 由 tests 自然判 unresolved）
            - build 失败                    -> metadata["build_failed"]=True（继续跑）
            - resolved = F2P/P2P 全过 AND integrity_ok
        """
        verification_id = f"clean-{task.task_id}-{uuid.uuid4().hex[:8]}"
        try:
            ws = self.isolation.create(task)
        except Exception as e:
            # 主 reward 语义不变: error 也是 0.0（binary_reward 以 verdict 为准）
            return VerificationResult(
                verification_id=verification_id,
                task_id=task.task_id,
                patch=patch,
                verdict="error",
                integrity_ok=False,
                reward=0.0,
                metadata={
                    "verifier": VERIFIER_VERSION,
                    "isolation": type(self.isolation).__name__,
                    "error": f"isolation.create failed: {e}",
                    **(metadata_extra or {}),
                },
            )
        try:
            apply_error = self.isolation.apply_patch(ws.path, patch)
            integrity_ok = apply_error is None

            results, build_failed = self._run_build_and_tests(task, ws.path)
            f2p = [r for r in results if r.kind == "fail_to_pass"]
            p2p = [r for r in results if r.kind == "pass_to_pass"]
            resolved = (
                integrity_ok
                and bool(f2p)
                and all(r.passed for r in f2p)
                and all(r.passed for r in p2p)
            )
            metadata: dict = {
                "verifier": VERIFIER_VERSION,
                "isolation": type(self.isolation).__name__,
                "isolated": True,
                **(metadata_extra or {}),
            }
            if apply_error:
                metadata["apply_error"] = apply_error
            if build_failed:
                metadata["build_failed"] = True
            if ws.setup_failures:
                metadata["setup_failures"] = ws.setup_failures

            v = VerificationResult(
                verification_id=verification_id,
                task_id=task.task_id,
                patch=patch,
                verdict="resolved" if resolved else "unresolved",
                fail_to_pass=f2p,
                pass_to_pass=p2p,
                integrity_ok=integrity_ok,
                metadata=metadata,
            )
            # baseline reward（主 reward 语义, 训练侧可覆盖为 shaped）
            v.reward = binary_reward(v)
            return v
        finally:
            self.isolation.destroy(ws.path)

    def _run_build_and_tests(
        self, task: TaskSpec, workdir: Path
    ) -> tuple[list[TestResult], bool]:
        """build_commands 依次真实执行（失败标记并继续）; 然后逐条跑测试。

        子进程注入 _child_env（PYTHONDONTWRITEBYTECODE + PYTHONPATH=workdir,
        阶段 13 的 __pycache__/editable 免疫）。
        """
        env = _child_env(workdir)
        build_failed = False
        for cmd in task.environment.build_commands:
            obs = _run_shell(workdir, cmd, self.bash_timeout, env=env)
            if obs.exit_code != 0:
                build_failed = True
                break

        results: list[TestResult] = []
        specs = [*task.fail_to_pass, *task.pass_to_pass]
        for spec in specs:
            start = time.monotonic()
            output, exit_code = _run_argv(
                workdir, task.environment.test_commands.get(spec.test_id, []),
                timeout=self.bash_timeout, env=env,
            )
            results.append(
                TestResult(
                    test_id=spec.test_id,
                    kind=spec.kind,
                    passed=exit_code == 0,
                    duration_ms=int((time.monotonic() - start) * 1000),
                    output=output[-_OUTPUT_CAP:] or None,
                )
            )
        return results, build_failed


# ------------------------- 内部工具（与 Mock 语义一致） -------------------------


def _child_env(workdir: Path | str) -> dict[str, str]:
    """子进程环境: 禁写字节码 + PYTHONPATH 指向工作目录。

    与 data/factory/verify.py 的免疫同源（阶段 13 坑）: 变异/checkout 与
    pytest 同秒时 __pycache__ 秒级 mtime 校验会误用旧 pyc; PYTHONPATH
    保证 import 命中工作树源码而非 site-packages 的 editable 残留。
    """
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(workdir) + (os.pathsep + existing if existing else "")
    return env


def _run_shell(cwd: Path, command: str, timeout: float, env: dict[str, str] | None = None):
    """真实执行 shell 命令（setup/build 用, shell 字符串语义）; 超时 -> 124。"""
    try:
        proc = subprocess.run(
            command, shell=True, cwd=cwd, capture_output=True, text=True,
            timeout=timeout, env=env or os.environ.copy(),
        )
        return _Obs(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except subprocess.TimeoutExpired as e:
        return _Obs(
            exit_code=124, stdout=e.stdout or "", stderr=(e.stderr or "") + "\n[timed out]"
        )


def _run_argv(
    cwd: Path, argv: list[str], timeout: float, env: dict[str, str] | None = None
) -> tuple[str, int]:
    """执行一条 argv 命令（TaskSpec.test_commands 的值 = 单条命令的参数列表）。

    无 shell; 超时 -> exit 124 + [timed out]。
    """
    try:
        proc = subprocess.run(
            argv, shell=False, cwd=cwd, capture_output=True, text=True,
            timeout=timeout, env=env or os.environ.copy(),
        )
        return proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        return (e.stdout or "") + (e.stderr or "") + "\n[timed out]", 124
    except FileNotFoundError as e:
        return f"[command not found: {e}]", 127


class _Obs:
    __slots__ = ("exit_code", "stdout", "stderr")

    def __init__(self, exit_code: int, stdout: str, stderr: str):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
