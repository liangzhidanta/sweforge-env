"""Execution backends: LocalExecutor (dev/test) and DockerExecutor (gated, M2 wiring)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from sweforge.env_server.docker.manager import ContainerLimits, ROLE_TASK, container_labels
from sweforge.env_server.docker.path_policy import PathPolicy


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    truncated: bool = False


class Executor(Protocol):
    root: Path
    path_policy: PathPolicy
    max_output_chars: int
    max_view_lines: int

    def run_argv(self, argv: Sequence[str], timeout: float, cwd: str = ".",
                 input_text: str | None = None) -> CommandResult: ...
    def run_shell(self, command: str, timeout: float, cwd: str = ".") -> CommandResult: ...
    def read_text(self, path: str) -> str: ...
    def write_text(self, path: str, content: str) -> None: ...
    def close(self) -> None: ...


def _truncate(value: str, max_chars: int) -> tuple[str, bool]:
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars], True


class LocalExecutor:
    """Run commands and file ops directly on a temp workspace directory."""

    def __init__(
        self,
        root: Path,
        protected_paths: Sequence[str] = (),
        max_output_chars: int = 20_000,
        max_view_lines: int = 200,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path_policy = PathPolicy(self.root, protected_paths)
        self.max_output_chars = max_output_chars
        self.max_view_lines = max_view_lines
        self._temporary: tempfile.TemporaryDirectory[str] | None = None

    @classmethod
    def from_snapshot(
        cls, snapshot: Path, protected_paths: Sequence[str] = (),
        max_output_chars: int = 20_000, max_view_lines: int = 200,
    ) -> "LocalExecutor":
        temporary = tempfile.TemporaryDirectory(prefix="sweforge-env-")
        destination = Path(temporary.name) / "repo"
        shutil.copytree(
            snapshot, destination,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
        )
        executor = cls(destination, protected_paths, max_output_chars, max_view_lines)
        executor._temporary = temporary
        return executor

    def run_argv(self, argv, timeout=30.0, cwd=".", input_text=None):
        working_directory = self.path_policy.resolve(cwd)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                tuple(argv), cwd=working_directory, input=input_text, capture_output=True,
                text=True, timeout=timeout, shell=False, check=False,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(124, "", "command timed out",
                                 int((time.monotonic() - started) * 1000), timed_out=True)
        stdout, stdout_cut = _truncate(completed.stdout, self.max_output_chars)
        stderr, stderr_cut = _truncate(completed.stderr, self.max_output_chars)
        return CommandResult(completed.returncode, stdout, stderr,
                             int((time.monotonic() - started) * 1000),
                             truncated=stdout_cut or stderr_cut)

    def run_shell(self, command, timeout=30.0, cwd="."):
        working_directory = self.path_policy.resolve(cwd)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command, cwd=working_directory, capture_output=True,
                text=True, timeout=timeout, shell=True, check=False,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(124, "", "command timed out",
                                 int((time.monotonic() - started) * 1000), timed_out=True)
        stdout, stdout_cut = _truncate(completed.stdout, self.max_output_chars)
        stderr, stderr_cut = _truncate(completed.stderr, self.max_output_chars)
        return CommandResult(completed.returncode, stdout, stderr,
                             int((time.monotonic() - started) * 1000),
                             truncated=stdout_cut or stderr_cut)

    def read_text(self, path: str) -> str:
        return self.path_policy.resolve(path).read_text(encoding="utf-8")

    def write_text(self, path: str, content: str) -> None:
        candidate = self.path_policy.resolve(path)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(content, encoding="utf-8")

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __del__(self) -> None:
        self.close()


class DockerExecutor:
    """One non-root, networkless container per env. Docker daemon required (M2 wiring)."""

    def __init__(
        self, image: str, container_name: str, task_id: str, env_id: str,
        workspace: str = "/workspace", docker_binary: str = "docker",
        protected_paths: Sequence[str] = (), max_output_chars: int = 20_000,
        max_view_lines: int = 200, limits: ContainerLimits | None = None,
    ) -> None:
        self.image = image
        self.container_name = container_name
        self.workspace = workspace
        self.root = Path(workspace)
        self.path_policy = PathPolicy(self.root, protected_paths)
        self.docker_binary = docker_binary
        self.max_output_chars = max_output_chars
        self.max_view_lines = max_view_lines
        self.limits = limits or ContainerLimits()
        self._task_id = task_id
        self._env_id = env_id
        self._started = False

    def create_command(self) -> tuple[str, ...]:
        return (
            self.docker_binary, "run", "--detach", "--rm",
            "--network", "none",
            "--cpus", str(self.limits.cpus),
            "--memory", self.limits.memory,
            "--pids-limit", str(self.limits.pids),
            "--user", "1000:1000",
            "--workdir", self.workspace,
            "--name", self.container_name,
            *(f"--label={label}" for label in container_labels(self._task_id, self._env_id, ROLE_TASK)),
            self.image, "sleep", "infinity",
        )

    def start(self) -> None:
        if self._started:
            return
        completed = subprocess.run(self.create_command(), capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"failed to start container: {completed.stderr.strip()}")
        self._started = True

    def _exec(self, argv, timeout, input_text=None, workdir=None, max_chars=None):
        self.start()
        command = [self.docker_binary, "exec", "-i"]
        if workdir:
            command += ["--workdir", workdir]
        command += [self.container_name, *argv]
        started = time.monotonic()
        try:
            completed = subprocess.run(command, input=input_text, capture_output=True,
                                       text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return CommandResult(124, "", "command timed out",
                                 int((time.monotonic() - started) * 1000), timed_out=True)
        limit = self.max_output_chars if max_chars is None else max_chars
        stdout, stdout_cut = _truncate(completed.stdout, limit)
        stderr, stderr_cut = _truncate(completed.stderr, limit)
        return CommandResult(completed.returncode, stdout, stderr,
                             int((time.monotonic() - started) * 1000),
                             truncated=stdout_cut or stderr_cut)

    def run_argv(self, argv, timeout=30.0, cwd=".", input_text=None):
        resolved_cwd = self.path_policy.resolve(cwd).as_posix()
        return self._exec(tuple(argv), timeout, input_text, workdir=resolved_cwd)

    def run_shell(self, command, timeout=30.0, cwd="."):
        resolved_cwd = self.path_policy.resolve(cwd).as_posix()
        return self._exec(("/bin/sh", "-c", command), timeout, workdir=resolved_cwd)

    def read_text(self, path: str) -> str:
        resolved = self.path_policy.resolve(path).as_posix()
        result = self._exec(("cat", resolved), timeout=10, max_chars=None)
        if result.exit_code != 0:
            raise OSError(result.stderr)
        return result.stdout

    def write_text(self, path: str, content: str) -> None:
        resolved = self.path_policy.resolve(path).as_posix()
        result = self._exec(("/bin/sh", "-c", f"cat > {resolved}"), timeout=10, input_text=content)
        if result.exit_code != 0:
            raise OSError(result.stderr)

    def close(self) -> None:
        if not self._started:
            return
        subprocess.run((self.docker_binary, "rm", "-f", self.container_name),
                       capture_output=True, text=True, check=False)
        self._started = False
