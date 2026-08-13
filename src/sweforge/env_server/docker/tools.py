"""Canonical 五工具执行: bash / search / view_file / str_replace / finish。

语义逐条镜像 vendored environment/mock.py（SFT 与 RL 的观察渲染一致）:
    view_file     "%6d\\t%s" 编号行渲染（编号后恰好一个分隔符, 其后全内容含缩进）
    search        file:line 匹配, 超过上限截断 truncated=True
    str_replace   唯一匹配替换; empty old_string = 文件开头插入 / 创建文件
    finish        返回 FinishObservation(patch=export_patch)

唯一的后端差别: 操作走 Executor 抽象（LocalExecutor / DockerExecutor）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from sweforge.env_server.docker.executors import Executor
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

#: search 返回的最大匹配数（超出截断, 与 R2E-Gym 观察一致）
_MAX_SEARCH_MATCHES = 200
#: bash 默认超时（秒）
_DEFAULT_BASH_TIMEOUT = 60.0

#: 与 path_policy.PathPolicy.resolve 拒绝绝对路径的文案保持一致
_MUST_BE_RELATIVE = "path must be relative to the task workspace"


def _relpath_or_none(path: str, workspace_root: Path) -> str | None:
    """绝对路径 -> 工作区内相对路径; 工作区外返回 None（保持原拒绝语义）。

    相对路径原样返回（契约不变）。绝对路径在工作区内（如 /workspace/a.py）
    归一为相对路径, 进入工作区后由 PathPolicy 以 root 为基准解析; 工作区外
    （如 /app/...、/home/runner/...）返回 None -> 沿用 "must be relative" 错误,
    模型由此学不到错误前缀, 只会学到工作区内的写法。
    """
    p = Path(path)
    if not p.is_absolute():
        return path
    try:
        return str(p.relative_to(workspace_root))
    except ValueError:
        return None


def execute_action(
    executor: Executor,
    action: ToolAction,
    export_patch: Callable[[], str] | None = None,
) -> ToolObservation:
    """执行一条 canonical action, 返回结构化 observation（mirror mock.execute）。"""
    if isinstance(action, BashAction):
        return _bash(executor, action)
    if isinstance(action, SearchAction):
        return _search(executor, action)
    if isinstance(action, ViewFileAction):
        return _view_file(executor, action)
    if isinstance(action, StrReplaceAction):
        return _str_replace(executor, action)
    if isinstance(action, FinishAction):
        patch = export_patch() if export_patch is not None else None
        return FinishObservation(patch=patch)
    raise TypeError(f"unknown action: {type(action)}")


def _bash(executor: Executor, action: BashAction) -> BashObservation:
    result = executor.run_shell(action.command, timeout=_DEFAULT_BASH_TIMEOUT)
    return BashObservation(
        exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr,
        duration_ms=result.duration_ms,
    )


def _search(executor: Executor, action: SearchAction) -> SearchObservation:
    try:
        pattern = re.compile(action.query)
    except re.error as e:
        return SearchObservation(matches=[], error=f"invalid regex: {e}")

    if action.path:
        rel = _relpath_or_none(action.path, executor.root)
        if rel is None:
            return SearchObservation(matches=[], error=_MUST_BE_RELATIVE)
        try:
            kind = executor.path_stat(rel)
        except ValueError as e:
            return SearchObservation(matches=[], error=str(e))
        if kind is None:
            return SearchObservation(matches=[], error=f"path not found: {action.path}")
        if kind == "file":
            files = [rel]
        else:
            files = [child for child in executor.list_files(rel) if child]
    else:
        files = executor.list_files(".")

    matches: list[SearchMatch] = []
    for relative in files:
        try:
            text = executor.read_text(relative)
        except (OSError, UnicodeDecodeError):
            continue
        if "\x00" in text:  # 二进制文件跳过（等价 grep 对 binary 的行为）
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                matches.append(SearchMatch(path=relative, line=i, content=line.strip()))
                if len(matches) >= _MAX_SEARCH_MATCHES:
                    return SearchObservation(matches=matches, truncated=True)
    return SearchObservation(matches=matches)


def _view_file(executor: Executor, action: ViewFileAction) -> ViewFileObservation:
    rel = _relpath_or_none(action.path, executor.root)
    if rel is None:
        return ViewFileObservation(
            path=action.path, start_line=0, end_line=0, content="", error=_MUST_BE_RELATIVE
        )
    try:
        kind = executor.path_stat(rel)
    except ValueError as e:
        return ViewFileObservation(
            path=action.path, start_line=0, end_line=0, content="", error=str(e)
        )
    if kind is None:
        return ViewFileObservation(
            path=action.path, start_line=0, end_line=0, content="",
            error=f"file not found: {action.path}",
        )
    if kind == "dir":
        return ViewFileObservation(
            path=action.path, start_line=0, end_line=0, content="",
            error=f"is a directory: {action.path}",
        )
    lines = executor.read_text(rel).splitlines()
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
            error=f"view range {action.start_line or 1}-{action.end_line or total} "
            f"out of bounds (file has {total} lines)",
        )
    content = "\n".join(
        f"{i:6d}\t{line}" for i, line in enumerate(lines, 1) if start <= i <= end
    )
    return ViewFileObservation(
        path=action.path, start_line=start, end_line=end, content=content, total_lines=total
    )


def _str_replace(executor: Executor, action: StrReplaceAction) -> StrReplaceObservation:
    rel = _relpath_or_none(action.path, executor.root)
    if rel is None:
        return StrReplaceObservation(success=False, path=action.path, error=_MUST_BE_RELATIVE)
    if action.old_string == "":
        # empty old_string: 在文件开头插入 new_string; 文件不存在则以 new_string 创建
        try:
            existing = executor.read_text(rel)
        except ValueError as e:
            return StrReplaceObservation(success=False, path=action.path, error=str(e))
        except OSError:
            existing = ""
        try:
            executor.write_text(rel, action.new_string + existing)
        except OSError as e:
            return StrReplaceObservation(success=False, path=action.path, error=str(e))
        return StrReplaceObservation(success=True, path=action.path)
    try:
        content = executor.read_text(rel)
    except ValueError as e:
        return StrReplaceObservation(success=False, path=action.path, error=str(e))
    except OSError:
        return StrReplaceObservation(
            success=False, path=action.path, error=f"file not found: {action.path}"
        )
    count = content.count(action.old_string)
    if count == 0:
        return StrReplaceObservation(success=False, path=action.path, error="old_string not found")
    if count > 1:
        return StrReplaceObservation(
            success=False, path=action.path,
            error=f"old_string not unique ({count} occurrences)",
        )
    try:
        executor.write_text(rel, content.replace(action.old_string, action.new_string))
    except OSError as e:
        return StrReplaceObservation(success=False, path=action.path, error=str(e))
    return StrReplaceObservation(success=True, path=action.path)
