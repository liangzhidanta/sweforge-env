"""LocalDockerBackend — create/reset/execute/export_patch/verify/destroy for one env."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from sweforge.env_server.docker.executors import Executor, LocalExecutor
from sweforge.env_server.docker.tools import execute_action
from sweforge.schemas import Observation, TaskSpec, ToolAction, VerificationResult


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
    workspace: Path
    closed: bool = False


class EnvironmentBackend(Protocol):
    def create(self, task: TaskSpec) -> EnvHandle: ...
    def reset(self, env: EnvHandle) -> str: ...
    def execute(self, env: EnvHandle, action: ToolAction) -> Observation: ...
    def export_patch(self, env: EnvHandle) -> str: ...
    def verify(self, task: TaskSpec, patch: str) -> VerificationResult: ...
    def destroy(self, env: EnvHandle) -> None: ...


def load_task_bundle(bundle_dir: Path) -> TaskBundle:
    manifest = json.loads((bundle_dir / "task_manifest.json").read_text(encoding="utf-8"))
    task = TaskSpec.from_dict(manifest["task"])
    repo_path = bundle_dir / "repo"
    hidden_tests = bundle_dir / "private" / "hidden_tests"
    if not repo_path.is_dir():
        raise FileNotFoundError(f"bundle {bundle_dir} missing repo/")
    return TaskBundle(
        task=task, repo_path=repo_path, hidden_tests=hidden_tests,
        integrity_protected=tuple(manifest.get("integrity_protected", [])),
    )


def _git_init_workspace(executor: Executor) -> None:
    executor.run_argv(("git", "init", "-q"), timeout=30)
    executor.run_argv(("git", "config", "user.email", "sweforge@localhost"), timeout=10)
    executor.run_argv(("git", "config", "user.name", "SWE-Forge"), timeout=10)
    executor.run_argv(("git", "add", "-A"), timeout=60)
    executor.run_argv(("git", "commit", "-q", "-m", "baseline"), timeout=60)


def _patch_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[2].startswith("a/"):
                paths.append(parts[2][2:])
        elif line.startswith("--- a/"):
            paths.append(line[6:])
    return paths


def _check_integrity(patch: str, protected_paths: Sequence[str], max_patch_chars: int = 200_000) -> tuple[bool, str]:
    if len(patch) > max_patch_chars:
        return False, f"patch too large ({len(patch)} > {max_patch_chars})"
    for path in _patch_paths(patch):
        for protected in protected_paths:
            protected = protected.strip("/")
            if path == protected or path.startswith(protected + "/"):
                return False, f"patch touches protected path: {path}"
    return True, "ok"


def _inject_hidden_tests(fresh: LocalExecutor, hidden_tests: Path) -> None:
    if not hidden_tests.is_dir():
        return
    for source in sorted(hidden_tests.rglob("*.py")):
        relative = source.relative_to(hidden_tests).as_posix()
        fresh.write_text(relative, source.read_text(encoding="utf-8"))


def _run_tests(executor: Executor, test_command: Sequence[str], test_ids: Sequence[str],
               timeout: float = 120.0) -> tuple[dict[str, bool], bool]:
    outcomes: dict[str, bool] = {}
    timed_out = False
    for test_id in test_ids:
        result = executor.run_argv((*test_command, test_id), timeout=timeout)
        outcomes[test_id] = result.exit_code == 0 and not result.timed_out
        timed_out = timed_out or result.timed_out
    return outcomes, timed_out


def _verify_clean(fresh: LocalExecutor, task: TaskSpec, patch: str, bundle: TaskBundle) -> VerificationResult:
    _git_init_workspace(fresh)
    integrity_ok, integrity_reason = _check_integrity(patch, task.protected_paths)
    apply_result = None
    if integrity_ok:
        apply_result = fresh.run_argv(("git", "apply", "--allow-empty", "--whitespace=nowarn", "-"),
                                      timeout=30, input_text=patch)
        integrity_ok = apply_result.exit_code == 0
    if integrity_ok:
        _inject_hidden_tests(fresh, bundle.hidden_tests)
    if not integrity_ok:
        f2p_outcomes = {test_id: False for test_id in task.fail_to_pass}
        p2p_outcomes = {test_id: False for test_id in task.pass_to_pass}
    else:
        f2p_outcomes, f2p_timeout = _run_tests(fresh, task.test_command, task.fail_to_pass)
        p2p_outcomes, p2p_timeout = _run_tests(fresh, task.test_command, task.pass_to_pass)
    timed_out = bool(apply_result and apply_result.timed_out) or locals().get("f2p_timeout", False) or locals().get("p2p_timeout", False)
    f2p_passed = sum(1 for ok in f2p_outcomes.values() if ok)
    f2p_total = len(f2p_outcomes)
    f2p_ratio = f2p_passed / f2p_total if f2p_total else 1.0
    p2p_passed = sum(1 for ok in p2p_outcomes.values() if ok)
    p2p_total = len(p2p_outcomes)
    p2p_ratio = p2p_passed / p2p_total if p2p_total else 1.0
    resolved = integrity_ok and f2p_ratio == 1.0 and p2p_ratio == 1.0 and not timed_out
    p2p_failure_rate = 0.0 if p2p_total == 0 else 1.0 - p2p_ratio
    reward = f2p_ratio - 0.3 * p2p_failure_rate + (0.2 if resolved else 0.0)
    return VerificationResult(
        f2p_passed=f2p_passed, f2p_total=f2p_total, f2p_ratio=f2p_ratio,
        p2p_passed=p2p_passed, p2p_total=p2p_total, p2p_ratio=p2p_ratio,
        integrity_ok=integrity_ok, resolved=resolved, reward=round(reward, 4), timeout=timed_out,
        details={"integrity": integrity_reason, "f2p": dict(f2p_outcomes), "p2p": dict(p2p_outcomes)},
    )


class LocalDockerBackend:
    def __init__(self, bundles_dir: Path, use_docker: bool = False) -> None:
        self.bundles_dir = Path(bundles_dir)
        self.use_docker = use_docker

    def _make_executor(self, task: TaskSpec) -> Executor:
        bundle = load_task_bundle(self.bundles_dir / task.task_id)
        if self.use_docker:
            raise NotImplementedError("DockerExecutor wiring lands in M2")
        executor = LocalExecutor.from_snapshot(bundle.repo_path, protected_paths=task.protected_paths)
        _git_init_workspace(executor)
        return executor

    def create(self, task: TaskSpec) -> EnvHandle:
        bundle = load_task_bundle(self.bundles_dir / task.task_id)
        executor = self._make_executor(task)
        return EnvHandle(env_id=uuid.uuid4().hex[:12], task=bundle.task,
                         executor=executor, workspace=executor.root)

    def reset(self, env: EnvHandle) -> str:
        env.executor.close()
        env.executor = self._make_executor(env.task)
        env.workspace = env.executor.root
        env.closed = False
        return env.task.problem_statement

    def execute(self, env: EnvHandle, action: ToolAction) -> Observation:
        if env.closed:
            raise RuntimeError("env is destroyed")
        return execute_action(env.executor, action, env.env_id)

    def export_patch(self, env: EnvHandle) -> str:
        if env.closed:
            raise RuntimeError("env is destroyed")
        env.executor.run_argv(("git", "add", "-N", "."), timeout=30)
        result = env.executor.run_argv(("git", "diff", "--no-ext-diff", "--", "."), timeout=30)
        if result.timed_out:
            raise TimeoutError("git diff timed out")
        return result.stdout

    def destroy(self, env: EnvHandle) -> None:
        if env.closed:
            return
        env.executor.close()
        env.closed = True

    def verify(self, task: TaskSpec, patch: str) -> VerificationResult:
        bundle = load_task_bundle(self.bundles_dir / task.task_id)
        fresh = LocalExecutor.from_snapshot(bundle.repo_path, protected_paths=task.protected_paths)
        try:
            return _verify_clean(fresh, task, patch, bundle)
        finally:
            fresh.close()
