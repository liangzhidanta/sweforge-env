"""Execution backends: LocalExecutor (dev/test) and DockerExecutor (gated, M2 wiring)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from sweforge.env_server.docker.manager import ContainerLimits, ROLE_TASK, container_labels
from sweforge.env_server.docker.path_policy import PathPolicy

#: run_argv/run_shell 的 capture_limit 哨兵值: 不截断（如导出完整 git diff）。
UNLIMITED = -1


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
                 input_text: str | None = None, env: dict[str, str] | None = None,
                 capture_limit: int | None = None) -> CommandResult: ...
    def run_shell(self, command: str, timeout: float, cwd: str = ".",
                  env: dict[str, str] | None = None,
                  capture_limit: int | None = None) -> CommandResult: ...
    def read_text(self, path: str) -> str: ...
    def write_text(self, path: str, content: str) -> None: ...
    def path_stat(self, path: str) -> str | None: ...
    def list_files(self, path: str = ".") -> list[str]: ...
    def search_text(self, pattern: str, path: str, max_results: int) -> list[str]: ...
    def close(self) -> None: ...


def _truncate(value: str, max_chars: int) -> tuple[str, bool]:
    if max_chars == UNLIMITED:
        return value, False
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars], True


def _limit(capture_limit: int | None, default_max: int) -> int:
    """capture_limit 归一: None -> 默认上限; UNLIMITED -> 不截断。"""
    return default_max if capture_limit is None else capture_limit


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

    def _env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        """Merge override env over the host environment (AutoDL _child_env semantics).

        Subprocess with env=replaces would drop PATH and break 'python' lookups;
        merging keeps the host PATH while letting callers override PYTHONPATH etc.
        DockerExecutor merges via --env=K=V flags, so the two backends agree.
        """
        if not env:
            return None
        merged = os.environ.copy()
        merged.update(env)
        return merged

    def run_argv(self, argv, timeout=30.0, cwd=".", input_text=None,
                 env=None, capture_limit=None):
        working_directory = self.path_policy.resolve(cwd)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                tuple(argv), cwd=working_directory, input=input_text, capture_output=True,
                text=True, timeout=timeout, shell=False, check=False, env=self._env(env),
            )
        except subprocess.TimeoutExpired:
            return CommandResult(124, "", "command timed out",
                                 int((time.monotonic() - started) * 1000), timed_out=True)
        limit = _limit(capture_limit, self.max_output_chars)
        stdout, stdout_cut = _truncate(completed.stdout, limit)
        stderr, stderr_cut = _truncate(completed.stderr, limit)
        return CommandResult(completed.returncode, stdout, stderr,
                             int((time.monotonic() - started) * 1000),
                             truncated=stdout_cut or stderr_cut)

    def run_shell(self, command, timeout=30.0, cwd=".", env=None, capture_limit=None):
        working_directory = self.path_policy.resolve(cwd)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command, cwd=working_directory, capture_output=True,
                text=True, timeout=timeout, shell=True, check=False, env=self._env(env),
            )
        except subprocess.TimeoutExpired:
            return CommandResult(124, "", "command timed out",
                                 int((time.monotonic() - started) * 1000), timed_out=True)
        limit = _limit(capture_limit, self.max_output_chars)
        stdout, stdout_cut = _truncate(completed.stdout, limit)
        stderr, stderr_cut = _truncate(completed.stderr, limit)
        return CommandResult(completed.returncode, stdout, stderr,
                             int((time.monotonic() - started) * 1000),
                             truncated=stdout_cut or stderr_cut)

    def list_files(self, path: str = ".") -> list[str]:
        root = self.path_policy.resolve(path)
        if root.is_file():
            return [root.relative_to(self.root).as_posix()]
        return sorted(
            p.relative_to(self.root).as_posix()
            for p in root.rglob("*")
            if p.is_file()
        )

    def read_text(self, path: str) -> str:
        return self.path_policy.resolve(path).read_text(encoding="utf-8")

    def write_text(self, path: str, content: str) -> None:
        candidate = self.path_policy.resolve(path)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(content, encoding="utf-8")

    def path_stat(self, path: str) -> str | None:
        resolved = self.path_policy.resolve(path)
        if resolved.is_file():
            return "file"
        if resolved.is_dir():
            return "dir"
        return None

    def search_text(self, pattern: str, path: str = ".", max_results: int = 50) -> list[str]:
        root = self.path_policy.resolve(path)
        expression = re.compile(pattern)
        results: list[str] = []
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for candidate in paths:
            if not candidate.is_file():
                continue
            try:
                relative = candidate.relative_to(self.root).as_posix()
                self.path_policy.resolve(relative)  # skip protected paths
                lines = candidate.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError, ValueError):
                continue
            for number, line in enumerate(lines, start=1):
                if expression.search(line):
                    results.append(f"{relative}:{number}:{line}")
                    if len(results) >= max_results:
                        return results
        return results

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

    def seed_from_snapshot(self, snapshot: Path) -> None:
        """Populate /workspace from a host snapshot dir (excludes bytecode/caches).

        Copies a tar into the container and extracts it inside so the workspace
        is owned by the container's uid 1000 user (docker cp alone would leave
        host ownership). Runs as the default container user, which can already
        write to /workspace (created + chowned in the base image).

        The archive is built with Python's tarfile module, NOT the host `tar`
        binary: macOS bsdtar synthesizes AppleDouble ``._*`` entries from host
        file xattrs (hidden even from its own ``-tf``) that the container's GNU
        tar materializes as real files, polluting the workspace.
        """
        self.start()
        with tempfile.NamedTemporaryFile(prefix="sweforge-seed-", suffix=".tar", delete=False) as handle:
            tar_path = handle.name
        try:
            with tarfile.open(tar_path, "w") as archive:
                for path in sorted(snapshot.rglob("*")):
                    if not path.is_file():
                        continue
                    if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
                        continue
                    if path.suffix == ".pyc":
                        continue
                    archive.add(path, arcname=path.relative_to(snapshot).as_posix())
            # NamedTemporaryFile 0600 -> docker cp 原样保留, 容器 uid 1000 读不了。
            os.chmod(tar_path, 0o644)
            self._copy_in(tar_path, "/tmp/sweforge-seed.tar")
            result = self._exec(
                ("/bin/sh", "-c", "tar -xf /tmp/sweforge-seed.tar -C /workspace"),
                timeout=120, max_chars=None,
            )
            if result.exit_code != 0:
                raise OSError(f"failed to extract workspace seed: {result.stderr.strip()}")
        finally:
            Path(tar_path).unlink(missing_ok=True)

    def _copy_in(self, host_path: str, container_path: str) -> None:
        completed = subprocess.run(
            (self.docker_binary, "cp", host_path, f"{self.container_name}:{container_path}"),
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            raise OSError(f"docker cp failed: {completed.stderr.strip()}")

    def _exec(self, argv, timeout, input_text=None, workdir=None, max_chars=None,
              env=None):
        self.start()
        command = [self.docker_binary, "exec", "-i"]
        if env:
            command += [f"--env={key}={value}" for key, value in env.items()]
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
        if max_chars is None:
            stdout, stdout_cut = completed.stdout, False
            stderr, stderr_cut = completed.stderr, False
        else:
            stdout, stdout_cut = _truncate(completed.stdout, max_chars)
            stderr, stderr_cut = _truncate(completed.stderr, max_chars)
        return CommandResult(completed.returncode, stdout, stderr,
                             int((time.monotonic() - started) * 1000),
                             truncated=stdout_cut or stderr_cut)

    def run_argv(self, argv, timeout=30.0, cwd=".", input_text=None, env=None,
                 capture_limit=None):
        resolved_cwd = self.path_policy.resolve(cwd).as_posix()
        return self._exec(tuple(argv), timeout, input_text, workdir=resolved_cwd,
                          max_chars=_limit(capture_limit, self.max_output_chars),
                          env=env)

    def run_shell(self, command, timeout=30.0, cwd=".", env=None, capture_limit=None):
        resolved_cwd = self.path_policy.resolve(cwd).as_posix()
        return self._exec(("/bin/sh", "-c", command), timeout, workdir=resolved_cwd,
                          max_chars=_limit(capture_limit, self.max_output_chars),
                          env=env)

    def list_files(self, path: str = ".") -> list[str]:
        root = self.path_policy.resolve(path)
        if root.is_file():
            return [root.relative_to(self.root).as_posix()]
        listing = self._exec(("find", root.as_posix(), "-type", "f"),
                             timeout=30, max_chars=UNLIMITED)
        if listing.exit_code != 0:
            return []
        return [
            line[len(str(self.root)) :].lstrip("/") if line.startswith(str(self.root))
            else line.lstrip("/")
            for line in listing.stdout.splitlines()
            if line
        ]

    def read_text(self, path: str) -> str:
        resolved = self.path_policy.resolve(path).as_posix()
        result = self._exec(("cat", resolved), timeout=10, max_chars=None)
        if result.exit_code != 0:
            raise OSError(result.stderr)
        return result.stdout

    def write_text(self, path: str, content: str) -> None:
        resolved = self.path_policy.resolve(path).as_posix()
        result = self._exec(("/bin/sh", "-c", f"cat > {resolved}"), timeout=10,
                            input_text=content, max_chars=self.max_output_chars)
        if result.exit_code != 0:
            raise OSError(result.stderr)

    def path_stat(self, path: str) -> str | None:
        resolved = self.path_policy.resolve(path).as_posix()
        if self._exec(("test", "-f", resolved), timeout=10).exit_code == 0:
            return "file"
        if self._exec(("test", "-d", resolved), timeout=10).exit_code == 0:
            return "dir"
        return None

    def search_text(self, pattern: str, path: str = ".", max_results: int = 50) -> list[str]:
        expression = re.compile(pattern)
        results: list[str] = []
        for relative in self.list_files(path):
            try:
                self.path_policy.resolve(relative)  # skip protected paths
                lines = self.read_text(relative).splitlines()
            except (UnicodeDecodeError, OSError, ValueError):
                continue
            for number, line in enumerate(lines, start=1):
                if expression.search(line):
                    results.append(f"{relative}:{number}:{line}")
                    if len(results) >= max_results:
                        return results
        return results

    def close(self) -> None:
        if not self._started:
            return
        subprocess.run((self.docker_binary, "rm", "-f", self.container_name),
                       capture_output=True, text=True, check=False)
        self._started = False
