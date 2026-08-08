"""Canonical tool execution: bash / search / view_file / str_replace -> Observation."""

from __future__ import annotations

import re
import time

from sweforge.env_server.docker.executors import Executor
from sweforge.schemas import Observation, ToolAction


def execute_action(executor: Executor, action: ToolAction, env_id: str) -> Observation:
    started = time.monotonic()
    tool = action.tool
    args = action.arguments
    try:
        if tool == "bash":
            return _bash(executor, args, action.request_id, env_id, started)
        if tool == "search":
            return _search(executor, args, action.request_id, env_id, started)
        if tool == "view_file":
            return _view_file(executor, args, action.request_id, env_id, started)
        if tool == "str_replace":
            return _str_replace(executor, args, action.request_id, env_id, started)
        return _error(action.request_id, env_id, tool, f"unknown tool: {tool}", started)
    except (ValueError, OSError, re.error) as error:
        return _error(action.request_id, env_id, tool, str(error), started)


def _error(request_id: str, env_id: str, tool: str, message: str, started: float) -> Observation:
    return Observation(
        request_id=request_id, env_id=env_id, tool=tool, exit_code=None,
        stderr=message, content=f"ERROR: {message}",
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _duration_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _bash(executor: Executor, args: dict, request_id: str, env_id: str, started: float) -> Observation:
    command = args.get("command", "")
    timeout = float(args.get("timeout", 30.0))
    if not command.strip():
        raise ValueError("command cannot be empty")
    result = executor.run_shell(command, timeout=timeout)
    return Observation(
        request_id=request_id, env_id=env_id, tool="bash",
        exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr,
        truncated=result.truncated, duration_ms=_duration_ms(started),
    )


def _bound(max_chars: int, value: str) -> tuple[str, bool]:
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars], True


def _search(executor: Executor, args: dict, request_id: str, env_id: str, started: float) -> Observation:
    pattern = args.get("pattern", "")
    path = args.get("path", ".")
    max_results = int(args.get("max_results", 50))
    if not pattern:
        raise ValueError("pattern cannot be empty")
    matches = executor.search_text(pattern, path, max_results)
    content = "\n".join(matches) if matches else "NO_MATCHES"
    content, truncated = _bound(executor.max_output_chars, content)
    return Observation(
        request_id=request_id, env_id=env_id, tool="search",
        exit_code=0 if matches else 1, content=content, truncated=truncated,
        duration_ms=_duration_ms(started),
    )


def _view_file(executor: Executor, args: dict, request_id: str, env_id: str, started: float) -> Observation:
    path = args.get("path", "")
    start_line = int(args.get("start_line", 1))
    end_line = int(args.get("end_line", executor.max_view_lines))
    if start_line < 1 or end_line < start_line:
        raise ValueError(f"invalid line range [{start_line}, {end_line}]")
    if end_line - start_line + 1 > executor.max_view_lines:
        raise ValueError(f"line range exceeds max_view_lines={executor.max_view_lines}")
    lines = executor.read_text(path).splitlines()
    selected = lines[start_line - 1 : end_line]
    rendered = "\n".join(f"{number:>5} | {line}" for number, line in enumerate(selected, start=start_line))
    return Observation(
        request_id=request_id, env_id=env_id, tool="view_file",
        exit_code=0, content=rendered or "EMPTY_RANGE", duration_ms=_duration_ms(started),
    )


def _str_replace(executor: Executor, args: dict, request_id: str, env_id: str, started: float) -> Observation:
    path = args.get("path", "")
    old = args.get("old", "")
    new = args.get("new", "")
    expected = args.get("expected_occurrences")
    if not path:
        raise ValueError("path cannot be empty")
    if not old:
        raise ValueError("old cannot be empty")
    content = executor.read_text(path)
    occurrences = content.count(old)
    if occurrences == 0:
        raise ValueError(f"0 matches for old string in {path}")
    if expected is not None and int(expected) != occurrences:
        raise ValueError(f"expected {expected} occurrence(s), found {occurrences}")
    if expected is None and occurrences > 1:
        raise ValueError(f"multiple matches ({occurrences}); set expected_occurrences to disambiguate")
    executor.write_text(path, content.replace(old, new))
    return Observation(
        request_id=request_id, env_id=env_id, tool="str_replace",
        exit_code=0, content=f"replaced {occurrences} occurrence(s) in {path}",
        duration_ms=_duration_ms(started),
    )
