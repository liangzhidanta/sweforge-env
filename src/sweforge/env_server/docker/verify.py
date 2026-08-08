"""Executor-aware clean verifier for the Mac Docker path (P4/P5).

AutoDL's vendored CleanVerifier (reward/verifier.py) runs build/tests via host
subprocess, so it cannot isolate inside containers. This module re-implements
the *same* verification semantics on top of the Executor abstraction
(LocalExecutor / DockerExecutor):

    - isolation create failure      -> verdict="error" (verifier-side fault)
    - patch apply failure           -> integrity_ok=False + unresolved
    - setup failure                 -> metadata["setup_failures"] (continue,
                                        tests naturally resolve)
    - build failure                 -> metadata["build_failed"]=True (continue)
    - resolved = integrity_ok AND bool(f2p) AND all F2P AND all P2P
    - baseline reward               -> binary_reward (1.0/0.0, verdict-based)

The fresh workspace is seeded from the authoritative bundle snapshot (repo/ +
hidden tests), located by task_id by the caller, so a spoofed caller TaskSpec
cannot weaken integrity.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable, Sequence

from sweforge.env_server.docker.executors import Executor
from sweforge.schemas.task import TaskSpec
from sweforge.schemas.verification import (
    TestResult,
    VerificationResult,
    binary_reward,
)

__all__ = ["verify_clean", "VERIFIER_VERSION"]

VERIFIER_VERSION = "clean-v1"
_DEFAULT_BASH_TIMEOUT = 60.0
_OUTPUT_CAP = 2000
_MAX_PATCH_CHARS = 200_000


def verify_clean(
    task: TaskSpec,
    patch: str,
    *,
    make_executor: Callable[[], Executor],
    hidden_tests: Path | None = None,
    protected_paths: Sequence[str] = (),
    bash_timeout: float = _DEFAULT_BASH_TIMEOUT,
    metadata_extra: dict | None = None,
) -> VerificationResult:
    """Verify a patch in a fresh workspace created by make_executor().

    make_executor returns a *seeded* executor (repo snapshot copied into its
    root, no hidden tests, no setup yet). verify_clean injects hidden tests,
    runs setup, applies the patch, then runs build + tests.

    protected_paths: paths the patch must never touch (hidden test files etc.).
    A violation -> integrity_ok=False + unresolved, without applying the patch.
    """
    verification_id = f"clean-{task.task_id}-{uuid.uuid4().hex[:8]}"
    try:
        executor = make_executor()
    except Exception as e:
        return VerificationResult(
            verification_id=verification_id,
            task_id=task.task_id,
            patch=patch,
            verdict="error",
            integrity_ok=False,
            reward=0.0,
            metadata={
                "verifier": VERIFIER_VERSION,
                "isolation": "executor",
                "error": f"isolation.create failed: {e}",
                **(metadata_extra or {}),
            },
        )
    try:
        if hidden_tests is not None:
            _seed_hidden_tests(executor, hidden_tests)
        setup_failures = _run_setup(executor, task.environment.setup_commands, bash_timeout)

        integrity_ok, integrity_reason = _check_integrity(patch, protected_paths)
        apply_error = None
        if integrity_ok:
            apply_error = _apply_patch(executor, patch)
        integrity_ok = integrity_ok and apply_error is None

        results, build_failed = _run_build_and_tests(executor, task, bash_timeout)
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
            "isolation": "executor",
            "isolated": True,
            **(metadata_extra or {}),
        }
        if apply_error:
            metadata["apply_error"] = apply_error
        if not integrity_ok and integrity_reason != "ok":
            metadata["integrity_reason"] = integrity_reason
        if build_failed:
            metadata["build_failed"] = True
        if setup_failures:
            metadata["setup_failures"] = setup_failures

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
        v.reward = binary_reward(v)
        return v
    finally:
        executor.close()


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


def _check_integrity(patch: str, protected_paths: Sequence[str]) -> tuple[bool, str]:
    if len(patch) > _MAX_PATCH_CHARS:
        return False, f"patch too large ({len(patch)} > {_MAX_PATCH_CHARS})"
    for path in _patch_paths(patch):
        for protected in protected_paths:
            protected = protected.strip("/")
            if path == protected or path.startswith(protected + "/"):
                return False, f"patch touches protected path: {path}"
    return True, "ok"


def _seed_hidden_tests(executor: Executor, hidden_tests: Path) -> None:
    """Copy hidden test files into the executor root (relative layout preserved)."""
    if not hidden_tests.is_dir():
        return
    for source in sorted(hidden_tests.rglob("*.py")):
        relative = source.relative_to(hidden_tests).as_posix()
        executor.write_text(relative, source.read_text(encoding="utf-8"))


def _run_setup(executor: Executor, commands: list[str], bash_timeout: float) -> list[str]:
    failures: list[str] = []
    for cmd in commands:
        result = executor.run_shell(cmd, timeout=bash_timeout, env=_child_env(executor))
        if result.exit_code != 0:
            tail = (result.stderr or result.stdout)[-200:]
            failures.append(f"{cmd!r} -> exit {result.exit_code}: {tail}")
    return failures


def _apply_patch(executor: Executor, patch: str) -> str | None:
    """git init + git apply in the workspace. None = clean; str = error message."""
    if not patch.strip():
        return None  # empty patch is legal "no fix"; unresolved comes from tests
    if executor.run_argv(("git", "init", "-q"), timeout=30).exit_code != 0:
        return "git init failed"
    result = executor.run_argv(
        ("git", "apply", "-p1", "--whitespace=nowarn", "-"),
        timeout=60, input_text=patch,
    )
    if result.exit_code != 0:
        return (result.stderr or result.stdout).strip() or "git apply failed"
    return None


def _run_build_and_tests(
    executor: Executor, task: TaskSpec, bash_timeout: float
) -> tuple[list[TestResult], bool]:
    env = _child_env(executor)
    build_failed = False
    for cmd in task.environment.build_commands:
        result = executor.run_shell(cmd, timeout=bash_timeout, env=env)
        if result.exit_code != 0:
            build_failed = True
            break

    results: list[TestResult] = []
    for spec in [*task.fail_to_pass, *task.pass_to_pass]:
        argv = task.environment.test_commands.get(spec.test_id, [])
        start = time.monotonic()
        output, exit_code = _run_argv(executor, argv, bash_timeout, env)
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


def _child_env(executor: Executor) -> dict[str, str]:
    """Disable bytecode writes and point PYTHONPATH at the workspace tree."""
    return {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": executor.root.as_posix(),
    }


def _run_argv(
    executor: Executor, argv: list[str], timeout: float, env: dict[str, str]
) -> tuple[str, int]:
    """Run one argv command (test_commands value); timeout -> 124, missing -> 127."""
    if not argv:
        return "[no test command configured]", 1
    result = executor.run_argv(tuple(argv), timeout=timeout, env=env)
    if result.timed_out:
        return result.stdout + result.stderr + "\n[timed out]", 124
    return result.stdout + result.stderr, result.exit_code
