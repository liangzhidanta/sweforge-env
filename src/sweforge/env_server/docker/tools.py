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
        try:
            kind = executor.path_stat(action.path)
        except ValueError as e:
            return SearchObservation(matches=[], error=str(e))
        if kind is None:
            return SearchObservation(matches=[], error=f"path not found: {action.path}")
        if kind == "file":
            files = [action.path]
        else:
            files = [rel for rel in executor.list_files(action.path) if rel]
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
    try:
        kind = executor.path_stat(action.path)
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
    lines = executor.read_text(action.path).splitlines()
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
    if action.old_string == "":
        # empty old_string: 在文件开头插入 new_string; 文件不存在则以 new_string 创建
        try:
            existing = executor.read_text(action.path)
        except ValueError as e:
            return StrReplaceObservation(success=False, path=action.path, error=str(e))
        except OSError:
            existing = ""
        executor.write_text(action.path, action.new_string + existing)
        return StrReplaceObservation(success=True, path=action.path)
    try:
        content = executor.read_text(action.path)
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
    executor.write_text(action.path, content.replace(action.old_string, action.new_string))
    return StrReplaceObservation(success=True, path=action.path)
