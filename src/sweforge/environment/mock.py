"""阶段 5 MockEnvironmentBackend —— 本地临时目录上的真实文件语义。

与 Remote/LocalDocker 的唯一差别是"没有容器隔离"：五工具全部真实执行
（subprocess bash / 逐行正则 search / 编号行 view_file / 唯一匹配 str_replace），
view 渲染格式与 SFT 源数据一致（"%6d\\t%s"，编号后恰好一个分隔符，缩进保留），
保证 SFT 与 RL 轨迹中同一工具的观察渲染风格一致。

    create     -> 在 workspace_root 下建任务目录, 写入 seed_files, 执行
                  environment.setup_commands
    execute    -> 五工具语义（见各 _run_* 实现）
    export_patch -> 初始快照 vs 当前的 unified diff（difflib 纯 python,
                  不依赖 git; a/ b/ 前缀 + @@ hunk, 可被 git apply 消费）
    verify     -> mock 语义的 verifier: 全新临时目录写入初始快照 + git apply
                  patch + build_commands + test_commands 真实执行。真实应用
                  patch、真实跑命令, 不伪造 reward；容器隔离是阶段 9
                  clean verifier 的职责, mock 无隔离
    reset      -> 丢弃修改, 恢复初始快照（重跑 setup_commands）
    destroy    -> 删除工作区（幂等）

Agent Loop 只依赖 EnvironmentBackend 接口, 本类仅供本地开发/测试/CI 冒烟。
"""

from __future__ import annotations

import difflib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from sweforge.environment.base import EnvironmentBackend
from sweforge.protocol.tools import (
    BashAction,
    BashObservation,
    FinishAction,
    FinishObservation,
    SearchAction,
    SearchMatch,
    SearchObservation,
    StrReplaceAction,
    StrReplaceObservation,
    ToolAction,
    ToolObservation,
    ViewFileAction,
    ViewFileObservation,
)
from sweforge.schemas.task import TaskSpec
from sweforge.schemas.verification import VerificationResult

__all__ = ["MockEnvironment", "MockEnvironmentBackend"]

#: search 返回的最大匹配数（超出截断, 与 R2E-Gym 观察一致）
_MAX_SEARCH_MATCHES = 200
#: bash 默认超时（秒）
_DEFAULT_BASH_TIMEOUT = 60.0
#: str_replace 后 observation 提示新旧内容差异的字符上限（防御超大 diff）
_PATCH_LINE_CAP = 4096


class MockEnvironment:
    """Mock 后端的 env 句柄（接口对句柄类型不设假设, 后端私有）。"""

    __slots__ = ("task_id", "workspace", "initial", "setup_commands", "finished", "destroyed")

    def __init__(self, task_id: str, workspace: Path, initial: dict[str, str], setup_commands: list[str]):
        self.task_id = task_id
        self.workspace = workspace
        self.initial = initial  # 初始快照（含 setup 产物）: 相对路径(posix) -> 内容
        self.setup_commands = setup_commands
        self.finished = False
        self.destroyed = False


class MockEnvironmentBackend(EnvironmentBackend):
    """阶段 5 Mock 实现（见模块 docstring）。"""

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        seed_files: dict[str, str] | None = None,
        bash_timeout: float = _DEFAULT_BASH_TIMEOUT,
    ):
        """seed_files: 注入初始 repo 内容（相对路径 -> 文本）。路径含 '/' 自动建目录。"""
        self.workspace_root = Path(workspace_root) if workspace_root else Path(tempfile.mkdtemp(prefix="sweforge-mock-"))
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.seed_files = dict(seed_files or {})
        self.bash_timeout = bash_timeout

    # ------------------------- 生命周期 -------------------------

    def create(self, task: TaskSpec) -> MockEnvironment:
        ws = self.workspace_root / f"{task.repo}-{task.task_id[:12]}"
        if ws.exists():
            shutil.rmtree(ws)
        ws.mkdir(parents=True, exist_ok=True)
        _write_files(ws, self.seed_files)
        for cmd in task.environment.setup_commands:
            _run_shell(ws, cmd, timeout=_DEFAULT_BASH_TIMEOUT)
        # 基线 = setup 之后的完整状态（export_patch / verify 的 base 语义）
        env = MockEnvironment(
            task_id=task.task_id,
            workspace=ws,
            initial=_read_files(ws),
            setup_commands=list(task.environment.setup_commands),
        )
        return env

    def reset(self, env: MockEnvironment) -> None:
        if env.destroyed:
            raise RuntimeError(f"env {env.task_id} already destroyed")
        for child in env.workspace.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        _write_files(env.workspace, self.seed_files)
        for cmd in env.setup_commands:
            _run_shell(env.workspace, cmd, timeout=_DEFAULT_BASH_TIMEOUT)
        env.initial = _read_files(env.workspace)
        env.finished = False

    def destroy(self, env: MockEnvironment) -> None:
        if env.destroyed:
            return
        shutil.rmtree(env.workspace, ignore_errors=True)
        env.destroyed = True

    # ------------------------- 五工具执行 -------------------------

    def execute(self, env: MockEnvironment, action: ToolAction) -> ToolObservation:
        if env.destroyed:
            raise RuntimeError(f"env {env.task_id} already destroyed")
        if isinstance(action, BashAction):
            return self._bash(env, action)
        if isinstance(action, SearchAction):
            return self._search(env, action)
        if isinstance(action, ViewFileAction):
            return self._view_file(env, action)
        if isinstance(action, StrReplaceAction):
            return self._str_replace(env, action)
        if isinstance(action, FinishAction):
            env.finished = True
            return FinishObservation(patch=self.export_patch(env))
        raise TypeError(f"unknown action: {type(action)}")

    def _bash(self, env: MockEnvironment, action: BashAction) -> BashObservation:
        return _run_shell(env.workspace, action.command, timeout=self.bash_timeout)

    def _search(self, env: MockEnvironment, action: SearchAction) -> SearchObservation:
        try:
            pattern = re.compile(action.query)
        except re.error as e:
            return SearchObservation(matches=[], error=f"invalid regex: {e}")

        if action.path:
            target = env.workspace / action.path
            if not target.exists():
                return SearchObservation(matches=[], error=f"path not found: {action.path}")
            files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())
        else:
            files = sorted(p for p in env.workspace.rglob("*") if p.is_file())

        matches: list[SearchMatch] = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "\x00" in text:  # 二进制文件跳过（等价 grep 对 binary 的行为）
                continue
            rel = f.relative_to(env.workspace).as_posix()
            for i, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    matches.append(SearchMatch(path=rel, line=i, content=line.strip()))
                    if len(matches) >= _MAX_SEARCH_MATCHES:
                        return SearchObservation(matches=matches, truncated=True)
        return SearchObservation(matches=matches)

    def _view_file(self, env: MockEnvironment, action: ViewFileAction) -> ViewFileObservation:
        path = env.workspace / action.path
        if not path.exists():
            return ViewFileObservation(
                path=action.path, start_line=0, end_line=0, content="", error=f"file not found: {action.path}"
            )
        if path.is_dir():
            return ViewFileObservation(
                path=action.path, start_line=0, end_line=0, content="", error=f"is a directory: {action.path}"
            )
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        start = action.start_line or 1
        end = action.end_line or total
        if start < 1:
            start = 1
        if end > total:
            end = total
        if start > total or start > end:
            return ViewFileObservation(
                path=action.path, start_line=0, end_line=0, content="",
                error=f"view range {action.start_line or 1}-{action.end_line or total} out of bounds (file has {total} lines)",
            )
        # 编号行渲染: "%6d\t%s"（编号后恰好一个分隔符, 其后全是内容含缩进——
        # 与 R2E-Gym 源数据 / SFT 轨迹的 view 观察格式一致）
        content = "\n".join(
            f"{i:6d}\t{line}" for i, line in enumerate(lines, 1) if start <= i <= end
        )
        return ViewFileObservation(path=action.path, start_line=start, end_line=end, content=content, total_lines=total)

    def _str_replace(self, env: MockEnvironment, action: StrReplaceAction) -> StrReplaceObservation:
        path = env.workspace / action.path
        if action.old_string == "":
            # empty old_string: 在文件开头插入 new_string；文件不存在则以 new_string 创建
            existing = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(action.new_string + existing, encoding="utf-8")
            return StrReplaceObservation(success=True, path=action.path)
        if not path.exists():
            return StrReplaceObservation(success=False, path=action.path, error=f"file not found: {action.path}")
        content = path.read_text(encoding="utf-8", errors="replace")
        count = content.count(action.old_string)
        if count == 0:
            return StrReplaceObservation(success=False, path=action.path, error="old_string not found")
        if count > 1:
            return StrReplaceObservation(
                success=False, path=action.path, error=f"old_string not unique ({count} occurrences)"
            )
        path.write_text(content.replace(action.old_string, action.new_string), encoding="utf-8")
        return StrReplaceObservation(success=True, path=action.path)

    # ------------------------- patch / verify -------------------------

    def export_patch(self, env: MockEnvironment) -> str:
        """初始快照 vs 当前 -> unified diff（a/ b/ 前缀, @@ hunk, git apply 可消费）。"""
        current = _read_files(env.workspace)
        hunks: list[str] = []
        for rel in sorted(set(env.initial) | set(current)):
            old = env.initial.get(rel, "")
            new = current.get(rel, "")
            if old == new:
                continue
            diff = difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
            # 每行保留尾随 \n（git apply 要求 hunk 内容行以换行结束）
            hunks.append("".join(diff).rstrip("\n"))
        if not hunks:
            return ""
        return "\n".join(hunks) + "\n"

    def verify(self, task: TaskSpec, patch: str) -> VerificationResult:
        """Mock 语义 verifier: 委托 CleanVerifier（阶段 9, reward/verifier.py）。

        隔离 = TempDirIsolation（seed_files 基线, 全新临时目录, 真实执行
        build/test_commands）。不伪造 reward: 所有 TestResult 来自真实命令
        exit code; patch 应用失败 -> integrity_ok=False + unresolved。
        """
        from sweforge.reward.verifier import CleanVerifier, TempDirIsolation

        verifier = CleanVerifier(
            TempDirIsolation(seed_files=self.seed_files, bash_timeout=_DEFAULT_BASH_TIMEOUT),
            bash_timeout=_DEFAULT_BASH_TIMEOUT,
        )
        return verifier.verify(task, patch, metadata_extra={"backend": "mock"})


# ------------------------- 内部工具 -------------------------


def _write_files(workspace: Path, files: dict[str, str]) -> dict[str, str]:
    """写入文件集, 返回初始快照（posix 相对路径 -> 内容, 供 diff 基线）。"""
    snapshot: dict[str, str] = {}
    for rel, content in files.items():
        p = workspace / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        snapshot[Path(rel).as_posix()] = content
    return snapshot


def _read_files(workspace: Path) -> dict[str, str]:
    """工作区当前内容快照（跳过目录, 只读文本文件）。"""
    out: dict[str, str] = {}
    for p in workspace.rglob("*"):
        if p.is_file():
            try:
                out[p.relative_to(workspace).as_posix()] = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return out


def _run_shell(cwd: Path, command: str, timeout: float) -> BashObservation:
    try:
        proc = subprocess.run(
            command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return BashObservation(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except subprocess.TimeoutExpired as e:
        return BashObservation(
            exit_code=124, stdout=e.stdout or "", stderr=(e.stderr or "") + "\n[timed out]"
        )


# verify 的执行逻辑已上移 CleanVerifier（reward/verifier.py, 阶段 9）;
# Mock 侧仅保留 _run_shell（execute 的 bash 动作使用）。
