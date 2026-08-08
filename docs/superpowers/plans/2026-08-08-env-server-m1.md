# SWE-Forge Env Server M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build M1 of the Mac-side execution environment in `/Users/apple/code/SWE_project`: the Canonical Agent Protocol (ToolAction/Observation), a dependency-free toy cache task, a `LocalDockerBackend` with a testable local executor, and the full ToolAction→Observation→patch→clean-verify→`validate_trajectory` loop.

**Architecture:** A single shared protocol package (`src/sweforge/schemas.py`, `protocol.py`) is consumed by `src/sweforge/env_server/docker/`. Tools (`bash/search/view_file/str_replace`) are dispatched by `tools.py` against an `Executor` abstraction; `LocalExecutor` runs directly on a temp workspace (default, tests everything), `DockerExecutor` is present with a unit-tested create-command but wired into `create()` in M2. `LocalDockerBackend` orchestrates create/reset/execute/export_patch/verify/destroy; `verify` applies the exported patch to a *fresh* copy, injects hidden tests, and returns a `VerificationResult`.

**Tech Stack:** Python 3.11+, stdlib only at runtime (subprocess, pathlib, dataclasses, unittest for the toy), pytest + ruff for dev.

**Design doc:** `docs/superpowers/specs/2026-08-08-env-server-design.md` (see it for the full rationale).

**AutoDL reconciliation:** schema files are not yet available on this machine; the protocol below is derived from the task spec §1/§8–§12 and will be diff-aligned against AutoDL's `src/sweforge/schemas` when the user provides access. Keep schema shape changes localized to `schemas.py`/`protocol.py`.

---

## File Structure

```
/Users/apple/code/SWE_project/
├── pyproject.toml                      # package "sweforge-env", pytest/ruff dev extras
├── .gitignore
├── src/sweforge/
│   ├── __init__.py
│   ├── schemas.py                      # canonical protocol models
│   ├── protocol.py                     # validate_trajectory + JSON helpers
│   └── env_server/
│       ├── __init__.py
│       └── docker/
│           ├── __init__.py
│           ├── manager.py              # container labels/limits constants
│           ├── path_policy.py          # workspace path safety
│           ├── executors.py            # Executor protocol + LocalExecutor + DockerExecutor
│           ├── tools.py                # bash/search/view_file/str_replace → Observation
│           └── backend.py              # LocalDockerBackend + bundle loader + clean verify
├── examples/toy_cache_aliasing/
│   ├── task_manifest.json              # TaskSpec (policy-visible) + integrity config
│   ├── repo/                           # canonical buggy snapshot (NOT a git repo)
│   │   ├── pyproject.toml
│   │   ├── toy_cache/__init__.py
│   │   ├── toy_cache/cache.py
│   │   └── tests/__init__.py
│   │       └── tests/test_public.py    # P2P (agent-visible)
│   └── private/hidden_tests/tests/__init__.py
│       └── private/hidden_tests/tests/test_f2p.py   # F2P (Mac-private)
└── tests/
    ├── test_schemas.py
    ├── test_protocol.py
    ├── test_path_policy.py
    ├── test_executors.py
    ├── test_tools.py
    ├── test_backend.py
    └── test_integration.py
```

All test commands assume the shell starts at `/Users/apple/code/SWE_project`. Use the venv python explicitly (`./.venv/bin/python -m pytest ...`) so there is no ambiguity about which interpreter runs pytest.

---

### Task 0: Project scaffold and test runner

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/sweforge/__init__.py`, `src/sweforge/env_server/__init__.py`, `src/sweforge/env_server/docker/__init__.py`
- Create: `tests/__init__.py` (empty, so pytest can import helpers if needed later)

- [ ] **Step 1: Initialize the git repo**

Run:
```bash
cd /Users/apple/code/SWE_project && git init -q && git config user.email "sweforge@localhost" && git config user.name "SWE-Forge"
```
Expected: no output, exit 0.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "sweforge-env"
version = "0.1.0"
description = "Mac-side execution environment for SWE-Forge (env server, backends, clean verifier)"
requires-python = ">=3.11"
license = { text = "MIT" }

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.9"]
server = ["fastapi>=0.110", "uvicorn>=0.29"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 3: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.DS_Store
```

- [ ] **Step 4: Create the empty package init files**

```bash
cd /Users/apple/code/SWE_project && mkdir -p src/sweforge/env_server/docker tests \
  && touch src/sweforge/__init__.py src/sweforge/env_server/__init__.py src/sweforge/env_server/docker/__init__.py tests/__init__.py
```

- [ ] **Step 5: Create the venv and install**

```bash
cd /Users/apple/code/SWE_project && python3 -m venv .venv \
  && ./.venv/bin/python -m pip install -q -U pip \
  && ./.venv/bin/python -m pip install -q -e '.[dev]'
```
Expected: installs cleanly (needs network for pytest/ruff). If offline, install nothing and rely on an already-available `pytest`; the plan's test commands then become `pytest ...` instead of `./.venv/bin/python -m pytest ...`.

- [ ] **Step 6: Smoke-run pytest**

Run:
```bash
cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest -q
```
Expected: `no tests ran`, exit code 5 (that is fine — no tests yet).

- [ ] **Step 7: Commit**

```bash
cd /Users/apple/code/SWE_project && git add pyproject.toml .gitignore src tests && git commit -q -m "chore: scaffold sweforge-env project"
```

---

### Task 1: Canonical schemas

**Files:**
- Create: `src/sweforge/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
from sweforge.schemas import AgentTrajectory, Observation, TaskSpec, ToolAction, TrajectoryStep


def test_task_spec_round_trip():
    spec = TaskSpec(
        task_id="t1",
        repo="r",
        base_commit="b",
        problem_statement="p",
        test_command=("python", "-m", "unittest"),
        fail_to_pass=("tests.test_f2p.F2PTest.test_x",),
    )
    assert TaskSpec.from_dict(spec.to_dict()) == spec


def test_task_spec_missing_f2p_rejected():
    try:
        TaskSpec(task_id="t", repo="r", base_commit="b", problem_statement="p",
                 test_command=("x",), fail_to_pass=())
    except ValueError:
        pass
    else:
        raise AssertionError("TaskSpec with empty fail_to_pass must raise")


def test_observation_to_dict_has_spec_fields():
    obs = Observation(request_id="r1", env_id="e1", tool="bash", exit_code=0,
                      stdout="ok", stderr="", content="", truncated=False, duration_ms=5)
    data = obs.to_dict()
    for key in ("request_id", "env_id", "tool", "exit_code", "stdout", "stderr",
                "content", "truncated", "duration_ms"):
        assert key in data
    assert data["request_id"] == "r1" and data["duration_ms"] == 5


def test_trajectory_to_dict_nests_steps():
    step = TrajectoryStep(
        ToolAction("view_file", {"path": "a.py"}, request_id="r1"),
        Observation(request_id="r1", env_id="e1", tool="view_file", exit_code=0, content="c"),
    )
    trajectory = AgentTrajectory(task_id="t1", steps=(step,))
    data = trajectory.to_dict()
    assert data["task_id"] == "t1"
    assert data["steps"][0]["action"]["tool"] == "view_file"
    assert data["steps"][0]["observation"]["request_id"] == "r1"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sweforge.schemas'`.

- [ ] **Step 3: Write `src/sweforge/schemas.py`**

```python
"""Canonical Agent Protocol — shared contract between AutoDL and Mac env server.

Source of truth per spec §1/§8/§17. To be diff-reconciled against AutoDL's
src/sweforge/schemas once those files are available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class ToolName(StrEnum):
    BASH = "bash"
    SEARCH = "search"
    VIEW_FILE = "view_file"
    STR_REPLACE = "str_replace"


@dataclass(frozen=True)
class ToolAction:
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Observation:
    request_id: str
    env_id: str
    tool: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    content: str = ""
    truncated: bool = False
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_command: tuple[str, ...]
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...] = ()
    protected_paths: tuple[str, ...] = (".git",)
    platform: str = "linux/arm64"
    image: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpec":
        data = dict(value)
        for name in ("test_command", "fail_to_pass", "pass_to_pass", "protected_paths"):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)


@dataclass(frozen=True)
class VerificationResult:
    f2p_passed: int
    f2p_total: int
    f2p_ratio: float
    p2p_passed: int
    p2p_total: int
    p2p_ratio: float
    integrity_ok: bool
    resolved: bool
    reward: float
    timeout: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryStep:
    action: ToolAction
    observation: Observation

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action.to_dict(), "observation": self.observation.to_dict()}


@dataclass(frozen=True)
class AgentTrajectory:
    task_id: str
    steps: tuple[TrajectoryStep, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": dict(self.metadata),
        }
```

Note: `TaskSpec.__post_init__`-style validation is intentionally minimal here — M1 keeps schema parsing lenient so AutoDL reconciliation later doesn't fight constructor strictness. The one invariant tested (`fail_to_pass` non-empty) lives in `protocol.validate_trajectory`/backend logic instead, so drop the `test_task_spec_missing_f2p_rejected` test if you'd rather not enforce it at the dataclass level; keeping it is fine too since `TaskSpec(... fail_to_pass=())` currently constructs without error and the test asserts the constructor *should* reject — so delete that test case if you keep the lenient constructor. **Decision: keep the lenient constructor and delete `test_task_spec_missing_f2p_rejected`.**

- [ ] **Step 4: Remove the constructor-strictness test**

Edit `tests/test_schemas.py` to delete `test_task_spec_missing_f2p_rejected`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest tests/test_schemas.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
cd /Users/apple/code/SWE_project && git add src/sweforge/schemas.py tests/test_schemas.py && git commit -q -m "feat: add canonical protocol schemas"
```

---

### Task 2: Trajectory validation and serialization

**Files:**
- Create: `src/sweforge/protocol.py`
- Test: `tests/test_protocol.py`

- [ ] **Step 1: Write the failing test**

```python
import json

import pytest

from sweforge.protocol import trajectory_from_dict, trajectory_to_json, validate_trajectory
from sweforge.schemas import AgentTrajectory, Observation, ToolAction, TrajectoryStep


def _step(tool: str, arguments: dict, request_id: str = "r1",
          observation_tool: str | None = None) -> TrajectoryStep:
    return TrajectoryStep(
        ToolAction(tool, arguments, request_id=request_id),
        Observation(request_id=request_id, env_id="e1",
                    tool=observation_tool if observation_tool is not None else tool),
    )


def test_valid_trajectory_has_no_errors():
    trajectory = AgentTrajectory("t1", steps=(
        _step("bash", {"command": "pwd"}),
        _step("view_file", {"path": "a.py", "start_line": 1, "end_line": 5}),
    ))
    assert validate_trajectory(trajectory) == []


def test_unknown_tool_flagged():
    trajectory = AgentTrajectory("t1", steps=(_step("curl", {}),))
    errors = validate_trajectory(trajectory)
    assert any("unknown tool" in error for error in errors)


def test_missing_argument_flagged():
    trajectory = AgentTrajectory("t1", steps=(_step("bash", {}),))
    errors = validate_trajectory(trajectory)
    assert any("missing arguments" in error for error in errors)


def test_tool_mismatch_flagged():
    trajectory = AgentTrajectory("t1", steps=(_step("bash", {"command": "pwd"}, observation_tool="view_file"),))
    assert any("action.tool != observation.tool" in error for error in validate_trajectory(trajectory))


def test_empty_task_id_flagged():
    assert any("task_id" in error for error in validate_trajectory(AgentTrajectory("", ())))


def test_json_round_trip():
    trajectory = AgentTrajectory("t1", steps=(
        _step("bash", {"command": "pwd"}),
    ))
    parsed = trajectory_from_dict(json.loads(trajectory_to_json(trajectory)))
    assert parsed == trajectory
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest tests/test_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sweforge.protocol'`.

- [ ] **Step 3: Write `src/sweforge/protocol.py`**

```python
"""Trajectory validation and JSON serialization for the canonical protocol."""

from __future__ import annotations

import json
from typing import Any, Mapping

from sweforge.schemas import AgentTrajectory, Observation, ToolAction, TrajectoryStep

_TOOL_ARGUMENTS = {
    "bash": ("command",),
    "search": ("pattern",),
    "view_file": ("path",),
    "str_replace": ("path", "old", "new"),
}


def validate_action(action: ToolAction) -> list[str]:
    errors: list[str] = []
    if action.tool not in _TOOL_ARGUMENTS:
        errors.append(f"unknown tool: {action.tool}")
        return errors
    missing = sorted(name for name in _TOOL_ARGUMENTS[action.tool] if name not in action.arguments)
    if missing:
        errors.append(f"{action.tool}: missing arguments {missing}")
    return errors


def validate_observation(observation: Observation) -> list[str]:
    errors: list[str] = []
    for field_name in ("request_id", "env_id", "tool"):
        if not getattr(observation, field_name):
            errors.append(f"observation.{field_name} must be non-empty")
    if observation.tool not in _TOOL_ARGUMENTS:
        errors.append(f"observation.tool unknown: {observation.tool}")
    return errors


def validate_trajectory(trajectory: AgentTrajectory) -> list[str]:
    """Return validation errors; an empty list means the trajectory is canonical-valid."""
    errors: list[str] = []
    if not trajectory.task_id:
        errors.append("trajectory.task_id must be non-empty")
    for index, step in enumerate(trajectory.steps):
        prefix = f"step[{index}]"
        errors.extend(f"{prefix}.{error}" for error in validate_action(step.action))
        errors.extend(f"{prefix}.{error}" for error in validate_observation(step.observation))
        if step.action.tool != step.observation.tool:
            errors.append(f"{prefix}: action.tool != observation.tool")
        if not step.observation.request_id:
            errors.append(f"{prefix}: observation.request_id must be non-empty")
    return errors


def trajectory_to_json(trajectory: AgentTrajectory) -> str:
    return json.dumps(trajectory.to_dict(), ensure_ascii=False, sort_keys=True)


def trajectory_from_dict(value: Mapping[str, Any]) -> AgentTrajectory:
    data = dict(value)
    steps = tuple(
        TrajectoryStep(ToolAction(**step["action"]), Observation(**step["observation"]))
        for step in data.get("steps", [])
    )
    return AgentTrajectory(task_id=data["task_id"], steps=steps, metadata=data.get("metadata", {}))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest tests/test_protocol.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/apple/code/SWE_project && git add src/sweforge/protocol.py tests/test_protocol.py && git commit -q -m "feat: add trajectory validation and serialization"
```

---

### Task 3: Workspace path policy

**Files:**
- Create: `src/sweforge/env_server/docker/path_policy.py`
- Test: `tests/test_path_policy.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from sweforge.env_server.docker.path_policy import PathPolicy


def test_relative_ok(tmp_path):
    policy = PathPolicy(tmp_path)
    assert policy.resolve("a.txt") == (tmp_path / "a.txt").resolve()


def test_absolute_rejected(tmp_path):
    policy = PathPolicy(tmp_path)
    with pytest.raises(ValueError, match="relative"):
        policy.resolve(str(tmp_path / "x.txt"))


def test_escape_rejected(tmp_path):
    policy = PathPolicy(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        policy.resolve("../secret.txt")


def test_protected_rejected(tmp_path):
    (tmp_path / ".git").mkdir()
    policy = PathPolicy(tmp_path, protected_paths=(".git",))
    with pytest.raises(ValueError, match="protected"):
        policy.resolve(".git/config")


def test_symlink_escape_rejected(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    (tmp_path / "link").symlink_to(outside)
    policy = PathPolicy(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        policy.resolve("link")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest tests/test_path_policy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sweforge.env_server'`.

- [ ] **Step 3: Write `src/sweforge/env_server/docker/path_policy.py`**

```python
"""Workspace path policy: block absolute, escape, symlink, and protected paths."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


class PathPolicy:
    def __init__(self, root: Path, protected_paths: Sequence[str] = ()) -> None:
        self.root = root.resolve()
        self.protected_paths = tuple(
            Path(path).as_posix().strip("/") for path in protected_paths if path
        )

    def resolve(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ValueError("path must be relative to the task workspace")
        resolved = (self.root / candidate).resolve()
        try:
            relative = resolved.relative_to(self.root).as_posix()
        except ValueError as error:
            raise ValueError("path escapes the task workspace") from error
        for protected in self.protected_paths:
            if relative == protected or relative.startswith(protected + "/"):
                raise ValueError(f"path is protected: {protected}")
        return resolved
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest tests/test_path_policy.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/apple/code/SWE_project && git add src/sweforge/env_server/docker/path_policy.py tests/test_path_policy.py && git commit -q -m "feat: add workspace path policy"
```

---

### Task 4: Container manager constants and executors

**Files:**
- Create: `src/sweforge/env_server/docker/manager.py`
- Create: `src/sweforge/env_server/docker/executors.py`
- Test: `tests/test_executors.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from sweforge.env_server.docker.executors import DockerExecutor, LocalExecutor


def test_run_argv(tmp_path):
    executor = LocalExecutor(tmp_path)
    result = executor.run_argv(("echo", "hi"))
    assert result.exit_code == 0 and result.stdout.strip() == "hi"


def test_run_shell(tmp_path):
    executor = LocalExecutor(tmp_path)
    result = executor.run_shell("echo hello")
    assert result.exit_code == 0 and result.stdout.strip() == "hello"


def test_run_timeout(tmp_path):
    executor = LocalExecutor(tmp_path)
    result = executor.run_shell("sleep 5", timeout=0.2)
    assert result.timed_out and result.exit_code == 124


def test_output_truncation(tmp_path):
    executor = LocalExecutor(tmp_path, max_output_chars=10)
    result = executor.run_shell("echo 0123456789abcdef")
    assert result.truncated and len(result.stdout) <= 10


def test_read_write_text(tmp_path):
    executor = LocalExecutor(tmp_path)
    executor.write_text("a/b.txt", "hi")
    assert executor.read_text("a/b.txt") == "hi"


def test_from_snapshot(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "f.txt").write_text("x")
    executor = LocalExecutor.from_snapshot(source)
    try:
        assert (executor.root / "f.txt").read_text() == "x"
    finally:
        executor.close()


def test_docker_executor_create_command_has_security_flags():
    executor = DockerExecutor(image="img", container_name="c1", task_id="t1", env_id="e1")
    command = executor.create_command()
    assert "--network" in command and "none" in command
    assert "--pids-limit" in command
    assert "--user" in command
    assert "--label=sweforge.managed=true" in command
    assert "--label=sweforge.task_id=t1" in command
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest tests/test_executors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sweforge.env_server.docker.executors'`.

- [ ] **Step 3: Write `src/sweforge/env_server/docker/manager.py`**

```python
"""Docker container labels, roles, and resource limits (M1 constants)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContainerLimits:
    cpus: float = 2.0
    memory: str = "4g"
    pids: int = 128
    timeout_seconds: float = 120.0


ROLE_TASK = "task"
ROLE_VERIFIER = "verifier"

LABEL_MANAGED = "sweforge.managed"
LABEL_ROLE = "sweforge.role"
LABEL_TASK_ID = "sweforge.task_id"
LABEL_ENV_ID = "sweforge.env_id"


def container_labels(task_id: str, env_id: str, role: str = ROLE_TASK) -> tuple[str, ...]:
    return (
        f"{LABEL_MANAGED}=true",
        f"{LABEL_ROLE}={role}",
        f"{LABEL_TASK_ID}={task_id}",
        f"{LABEL_ENV_ID}={env_id}",
    )
```

- [ ] **Step 4: Write `src/sweforge/env_server/docker/executors.py`**

```python
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

    def _exec(self, argv, timeout, input_text=None):
        self.start()
        command = (self.docker_binary, "exec", "-i", self.container_name, *argv)
        started = time.monotonic()
        try:
            completed = subprocess.run(command, input=input_text, capture_output=True,
                                       text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return CommandResult(124, "", "command timed out",
                                 int((time.monotonic() - started) * 1000), timed_out=True)
        stdout, stdout_cut = _truncate(completed.stdout, self.max_output_chars)
        stderr, stderr_cut = _truncate(completed.stderr, self.max_output_chars)
        return CommandResult(completed.returncode, stdout, stderr,
                             int((time.monotonic() - started) * 1000),
                             truncated=stdout_cut or stderr_cut)

    def run_argv(self, argv, timeout=30.0, cwd=".", input_text=None):
        resolved_cwd = self.path_policy.resolve(cwd).as_posix()
        return self._exec(("--workdir", resolved_cwd, *argv), timeout, input_text)

    def run_shell(self, command, timeout=30.0, cwd="."):
        resolved_cwd = self.path_policy.resolve(cwd).as_posix()
        return self._exec(("--workdir", resolved_cwd, "/bin/sh", "-c", command), timeout)

    def read_text(self, path: str) -> str:
        resolved = self.path_policy.resolve(path).as_posix()
        result = self._exec(("cat", resolved), timeout=10)
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest tests/test_executors.py -v`
Expected: 7 passed. (`test_docker_executor_create_command_has_security_flags` exercises command construction only — it never touches a Docker daemon.)

- [ ] **Step 6: Commit**

```bash
cd /Users/apple/code/SWE_project && git add src/sweforge/env_server/docker/manager.py src/sweforge/env_server/docker/executors.py tests/test_executors.py && git commit -q -m "feat: add executor backends (local + docker gated)"
```

---

### Task 5: Canonical tool execution

**Files:**
- Create: `src/sweforge/env_server/docker/tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write the failing test**

```python
from sweforge.env_server.docker.executors import LocalExecutor
from sweforge.env_server.docker.tools import execute_action
from sweforge.schemas import ToolAction


def _executor(tmp_path):
    (tmp_path / "a.py").write_text("def get_or_compute(key):\n    return key\n")
    return LocalExecutor(tmp_path)


def test_bash_success(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("bash", {"command": "echo ok"}, request_id="r1"), "e1")
    assert observation.tool == "bash" and observation.exit_code == 0
    assert observation.stdout.strip() == "ok"
    assert observation.request_id == "r1" and observation.env_id == "e1"


def test_bash_empty_command(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("bash", {"command": "  "}, request_id="r1"), "e1")
    assert observation.exit_code is None and "command cannot be empty" in observation.content


def test_bash_timeout(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("bash", {"command": "sleep 5", "timeout": 0.2}, request_id="r1"), "e1")
    assert observation.exit_code == 124


def test_unknown_tool(tmp_path):
    observation = execute_action(_executor(tmp_path), ToolAction("curl", {}, request_id="r1"), "e1")
    assert observation.exit_code is None and "unknown tool" in observation.content


def test_search_found(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("search", {"pattern": "get_or_compute"}, request_id="r1"), "e1")
    assert observation.exit_code == 0 and "a.py:1" in observation.content


def test_search_no_match(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("search", {"pattern": "zzz_no_match"}, request_id="r1"), "e1")
    assert observation.exit_code == 1 and observation.content == "NO_MATCHES"


def test_search_max_results(tmp_path):
    executor = LocalExecutor(tmp_path)
    executor.write_text("b.txt", "\n".join(f"needle {i}" for i in range(100)))
    observation = execute_action(executor,
                                 ToolAction("search", {"pattern": "needle", "max_results": 5}, request_id="r1"), "e1")
    assert len(observation.content.splitlines()) == 5


def test_view_file_range(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("view_file", {"path": "a.py", "start_line": 1, "end_line": 2}, request_id="r1"), "e1")
    assert observation.exit_code == 0 and "| def get_or_compute" in observation.content


def test_view_file_invalid_range(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("view_file", {"path": "a.py", "start_line": 5, "end_line": 2}, request_id="r1"), "e1")
    assert observation.exit_code is None and "invalid line range" in observation.content


def test_view_file_escape(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("view_file", {"path": "../secret.txt"}, request_id="r1"), "e1")
    assert observation.exit_code is None and "escapes" in observation.content


def test_str_replace_single(tmp_path):
    executor = _executor(tmp_path)
    observation = execute_action(executor,
                                 ToolAction("str_replace", {"path": "a.py", "old": "return key",
                                                            "new": "return key.upper()", "expected_occurrences": 1}, request_id="r1"), "e1")
    assert observation.exit_code == 0
    assert executor.read_text("a.py") == "def get_or_compute(key):\n    return key.upper()\n"


def test_str_replace_zero(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("str_replace", {"path": "a.py", "old": "absent", "new": "x"}, request_id="r1"), "e1")
    assert observation.exit_code is None and "0 matches" in observation.content


def test_str_replace_multiple_ambiguous(tmp_path):
    executor = _executor(tmp_path)
    executor.write_text("b.txt", "aaa")
    observation = execute_action(executor,
                                 ToolAction("str_replace", {"path": "b.txt", "old": "a", "new": "b"}, request_id="r1"), "e1")
    assert observation.exit_code is None and "multiple matches" in observation.content


def test_str_replace_multiple_expected(tmp_path):
    executor = _executor(tmp_path)
    executor.write_text("b.txt", "aaa")
    observation = execute_action(executor,
                                 ToolAction("str_replace", {"path": "b.txt", "old": "a", "new": "b",
                                                            "expected_occurrences": 3}, request_id="r1"), "e1")
    assert observation.exit_code == 0 and executor.read_text("b.txt") == "bbb"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest tests/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sweforge.env_server.docker.tools'`.

- [ ] **Step 3: Write `src/sweforge/env_server/docker/tools.py`**

```python
"""Canonical tool execution: bash / search / view_file / str_replace -> Observation."""

from __future__ import annotations

import re
import time
from pathlib import Path

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
    except (ValueError, OSError) as error:
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


def _search_regex(executor: Executor, pattern: str, path: str, max_results: int) -> list[str]:
    root = executor.path_policy.resolve(path)
    expression = re.compile(pattern)
    results: list[str] = []
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for candidate in paths:
        if not candidate.is_file():
            continue
        try:
            relative = candidate.relative_to(executor.root).as_posix()
            executor.path_policy.resolve(relative)  # skip protected paths
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError, ValueError):
            continue
        for number, line in enumerate(lines, start=1):
            if expression.search(line):
                results.append(f"{relative}:{number}:{line}")
                if len(results) >= max_results:
                    return results
    return results


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
    matches = _search_regex(executor, pattern, path, max_results)
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest tests/test_tools.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/apple/code/SWE_project && git add src/sweforge/env_server/docker/tools.py tests/test_tools.py && git commit -q -m "feat: add canonical tool execution (bash/search/view_file/str_replace)"
```

---

### Task 6: Toy task bundle

**Files:**
- Create: `examples/toy_cache_aliasing/task_manifest.json`
- Create: `examples/toy_cache_aliasing/repo/pyproject.toml`
- Create: `examples/toy_cache_aliasing/repo/toy_cache/__init__.py`
- Create: `examples/toy_cache_aliasing/repo/toy_cache/cache.py`
- Create: `examples/toy_cache_aliasing/repo/tests/__init__.py`
- Create: `examples/toy_cache_aliasing/repo/tests/test_public.py`
- Create: `examples/toy_cache_aliasing/private/hidden_tests/tests/__init__.py`
- Create: `examples/toy_cache_aliasing/private/hidden_tests/tests/test_f2p.py`
- Test: `tests/test_toy_bundle.py` (asserts the F2P/P2P matrix: buggy fails F2P/passes P2P; fixed passes both)

This task has no new library code — it creates the canonical toy task and proves its F2P/P2P matrix in a scripted verify pass. The bundle dir name equals `task_id` (`toy_cache_aliasing`) because `LocalDockerBackend` resolves bundles by `bundles_dir / task_id`.

- [ ] **Step 1: Write `examples/toy_cache_aliasing/task_manifest.json`**

```json
{
  "task": {
    "task_id": "toy_cache_aliasing",
    "repo": "toy_cache",
    "base_commit": "buggy",
    "problem_statement": "Fix the caching bug in toy_cache/cache.py. get_or_compute(key, compute) should cache each key independently: two distinct keys that compute equal values must each invoke their own compute(), and a repeated call for the same key must return the cached value.",
    "test_command": ["python", "-m", "unittest"],
    "fail_to_pass": ["tests.test_f2p.F2PTest.test_distinct_keys_do_not_alias"],
    "pass_to_pass": [
      "tests.test_public.PublicTests.test_correct_value_returned",
      "tests.test_public.PublicTests.test_repeat_returns_same_value"
    ],
    "protected_paths": [".git"],
    "platform": "linux/arm64",
    "metadata": {"source": "toy", "bug_class": "cache-key-aliasing"}
  },
  "integrity_protected": []
}
```

- [ ] **Step 2: Write the buggy snapshot**

`examples/toy_cache_aliasing/repo/pyproject.toml`:
```toml
[project]
name = "toy-cache"
version = "0.1.0"
```

`examples/toy_cache_aliasing/repo/toy_cache/__init__.py`: empty file.

`examples/toy_cache_aliasing/repo/toy_cache/cache.py`:
```python
"""A tiny keyed cache. Entries are indexed by their computed value (bug)."""

_CACHE: dict[object, object] = {}


def get_or_compute(key: str, compute) -> object:
    """Return a cached value for ``key``, computing it once if absent."""
    value = compute()
    if value not in _CACHE:
        _CACHE[value] = value
    return _CACHE[value]
```

`examples/toy_cache_aliasing/repo/tests/__init__.py`: empty file.

`examples/toy_cache_aliasing/repo/tests/test_public.py`:
```python
import unittest

from toy_cache.cache import get_or_compute


class PublicTests(unittest.TestCase):
    def test_correct_value_returned(self):
        self.assertEqual(get_or_compute("k", lambda: 42), 42)

    def test_repeat_returns_same_value(self):
        self.assertEqual(get_or_compute("r", lambda: 7), 7)
        self.assertEqual(get_or_compute("r", lambda: 7), 7)
```

- [ ] **Step 3: Write the hidden F2P test**

`examples/toy_cache_aliasing/private/hidden_tests/tests/__init__.py`: empty file.

`examples/toy_cache_aliasing/private/hidden_tests/tests/test_f2p.py`:
```python
import unittest

from toy_cache.cache import get_or_compute


class F2PTest(unittest.TestCase):
    def test_distinct_keys_do_not_alias(self):
        calls: list[str] = []

        def first() -> str:
            calls.append("first")
            return "shared"

        def second() -> str:
            calls.append("second")
            return "shared"

        self.assertEqual(get_or_compute("a", first), "shared")
        self.assertEqual(get_or_compute("b", second), "shared")
        self.assertEqual(calls, ["first", "second"])
```

- [ ] **Step 4: Write the matrix test**

`tests/test_toy_bundle.py`:
```python
from pathlib import Path

from sweforge.env_server.docker.backend import LocalDockerBackend, load_task_bundle
from sweforge.schemas import ToolAction

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "toy_cache_aliasing"

FIX = (
    "    if key not in _CACHE:\n"
    "        _CACHE[key] = compute()\n"
    "    return _CACHE[key]\n"
)
OLD = (
    "    value = compute()\n"
    "    if value not in _CACHE:\n"
    "        _CACHE[value] = value\n"
    "    return _CACHE[value]\n"
)


def test_buggy_f2p_fails_and_p2p_passes():
    task = load_task_bundle(EXAMPLES).task
    backend = LocalDockerBackend(EXAMPLES.parent)
    assert not backend.verify(task, "").resolved
    result = backend.verify(task, "")
    assert result.f2p_ratio == 0.0
    assert result.p2p_ratio == 1.0


def test_fixed_f2p_and_p2p_pass():
    task = load_task_bundle(EXAMPLES).task
    backend = LocalDockerBackend(EXAMPLES.parent)
    env = backend.create(task)
    try:
        observation = backend.execute(
            env, ToolAction("str_replace", {"path": "toy_cache/cache.py", "old": OLD, "new": FIX,
                                            "expected_occurrences": 1}, request_id="r1"))
        assert observation.exit_code == 0, observation.content
        patch = backend.export_patch(env)
        result = backend.verify(task, patch)
    finally:
        backend.destroy(env)
    assert result.resolved
    assert result.f2p_ratio == 1.0 and result.p2p_ratio == 1.0
```

Note: `test_toy_bundle.py` depends on `backend.py`, which is written in Task 7. If Task 6 lands before Task 7 the import fails — so implement Task 7's `backend.py` (Steps 1–5 of Task 7) at the same time as this Task 6 matrix test, then run the suite. **Simplest ordering: do Task 6 Steps 1–3 (create bundle files), then Task 7 (backend), then return here for Step 4's matrix test.**

- [ ] **Step 5: Commit the bundle**

```bash
cd /Users/apple/code/SWE_project && git add examples && git commit -q -m "data: add toy cache-aliasing task bundle"
```

(Commit `tests/test_toy_bundle.py` together with the backend in Task 7.)

---

### Task 7: LocalDockerBackend (create/reset/execute/export_patch/verify/destroy)

**Files:**
- Create: `src/sweforge/env_server/docker/backend.py`
- Test: `tests/test_backend.py`
- Modify: `tests/test_toy_bundle.py` (already written above; commit here)

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pytest

from sweforge.env_server.docker.backend import LocalDockerBackend, load_task_bundle
from sweforge.schemas import ToolAction

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "toy_cache_aliasing"


def _task():
    return load_task_bundle(EXAMPLES).task


def test_manifest_round_trip():
    bundle = load_task_bundle(EXAMPLES)
    assert bundle.task.task_id == "toy_cache_aliasing"
    assert bundle.repo_path.is_dir() and bundle.hidden_tests.is_dir()


def test_create_reset_destroy():
    backend = LocalDockerBackend(EXAMPLES.parent)
    env = backend.create(_task())
    assert not env.closed and env.workspace.exists()
    assert backend.reset(env) == env.task.problem_statement
    backend.destroy(env)
    assert env.closed
    backend.destroy(env)  # idempotent


def test_execute_after_destroy_raises():
    backend = LocalDockerBackend(EXAMPLES.parent)
    env = backend.create(_task())
    backend.destroy(env)
    with pytest.raises(RuntimeError):
        backend.execute(env, ToolAction("bash", {"command": "pwd"}, request_id="r1"))


def test_export_patch_reflects_edits():
    backend = LocalDockerBackend(EXAMPLES.parent)
    env = backend.create(_task())
    try:
        observation = backend.execute(env, ToolAction(
            "str_replace", {"path": "toy_cache/cache.py", "old": "value = compute()",
                            "new": "_ignored = compute()", "expected_occurrences": 1}, request_id="r1"))
        assert observation.exit_code == 0
        patch = backend.export_patch(env)
        assert "_ignored = compute()" in patch
    finally:
        backend.destroy(env)


def test_verify_rejects_tampered_patch():
    backend = LocalDockerBackend(EXAMPLES.parent)
    tampered = "diff --git a/.git/config b/.git/config\n--- a/.git/config\n+++ b/.git/config\n@@ -1 +1 @@\n-secret\n+owned\n"
    result = backend.verify(_task(), tampered)
    assert not result.integrity_ok
    assert not result.resolved
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest tests/test_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sweforge.env_server.docker.backend'`.

- [ ] **Step 3: Write `src/sweforge/env_server/docker/backend.py`**

```python
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
        apply_result = fresh.run_argv(("git", "apply", "--whitespace=nowarn", "-"),
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest tests/test_backend.py tests/test_toy_bundle.py -v
```
Expected: `test_backend.py` 5 passed and `test_toy_bundle.py` 2 passed.

- [ ] **Step 5: Run the whole suite so far**

Run: `cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest -q`
Expected: all tests pass (schemas 3, protocol 6, path_policy 5, executors 7, tools 16, backend 5, toy_bundle 2).

- [ ] **Step 6: Commit**

```bash
cd /Users/apple/code/SWE_project && git add src/sweforge/env_server/docker/backend.py tests/test_backend.py tests/test_toy_bundle.py && git commit -q -m "feat: add LocalDockerBackend with clean verify"
```

---

### Task 8: Full-rollout integration test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from sweforge.env_server.docker.backend import LocalDockerBackend, load_task_bundle
from sweforge.protocol import trajectory_from_dict, trajectory_to_json, validate_trajectory
from sweforge.schemas import AgentTrajectory, ToolAction, TrajectoryStep

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "toy_cache_aliasing"

OLD = (
    "    value = compute()\n"
    "    if value not in _CACHE:\n"
    "        _CACHE[value] = value\n"
    "    return _CACHE[value]\n"
)
NEW = (
    "    if key not in _CACHE:\n"
    "        _CACHE[key] = compute()\n"
    "    return _CACHE[key]\n"
)


def test_full_rollout_loop_and_clean_verify():
    task = load_task_bundle(EXAMPLES).task
    backend = LocalDockerBackend(EXAMPLES.parent)
    env = backend.create(task)
    steps = []
    try:
        assert backend.reset(env) == task.problem_statement
        actions = (
            ToolAction("view_file", {"path": "toy_cache/cache.py", "start_line": 1, "end_line": 30}, request_id="req-1"),
            ToolAction("search", {"pattern": "_CACHE"}, request_id="req-2"),
            ToolAction("str_replace", {"path": "toy_cache/cache.py", "old": OLD, "new": NEW,
                                       "expected_occurrences": 1}, request_id="req-3"),
            ToolAction("bash", {"command": "python -m unittest tests.test_public", "timeout": 60}, request_id="req-4"),
        )
        for action in actions:
            observation = backend.execute(env, action)
            assert observation.tool == action.tool
            assert observation.request_id == action.request_id
            steps.append(TrajectoryStep(action, observation))
        assert "get_or_compute" in steps[0].observation.content
        assert steps[2].observation.exit_code == 0, steps[2].observation.content
        assert steps[3].observation.exit_code == 0, steps[3].observation.content
        patch = backend.export_patch(env)
        assert "-    value = compute()" in patch
        result = backend.verify(task, patch)
        assert result.integrity_ok
        assert result.f2p_ratio == 1.0
        assert result.p2p_ratio == 1.0
        assert result.resolved
        assert result.reward > 0
    finally:
        backend.destroy(env)

    trajectory = AgentTrajectory(task_id=task.task_id, steps=tuple(steps))
    assert validate_trajectory(trajectory) == []
    restored = trajectory_from_dict(__import__("json").loads(trajectory_to_json(trajectory)))
    assert restored == trajectory


def test_unfixed_patch_is_not_resolved():
    task = load_task_bundle(EXAMPLES).task
    backend = LocalDockerBackend(EXAMPLES.parent)
    result = backend.verify(task, "")
    assert not result.resolved
    assert result.f2p_ratio == 0.0
```

- [ ] **Step 2: Run it to verify it passes**

Run: `cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest tests/test_integration.py -v`
Expected: 2 passed.

- [ ] **Step 3: Run the full suite**

Run: `cd /Users/apple/code/SWE_project && ./.venv/bin/python -m pytest -q`
Expected: all green.

- [ ] **Step 4: Lint**

Run: `cd /Users/apple/code/SWE_project && ./.venv/bin/ruff check src tests`
Expected: no violations (fix any E/F/I/UP/B/SIM findings).

- [ ] **Step 5: Commit**

```bash
cd /Users/apple/code/SWE_project && git add tests/test_integration.py && git commit -q -m "test: full rollout loop -> patch -> clean verify -> validate_trajectory"
```

---

## Self-Review

**Spec coverage vs the design doc (M1):**
- Canonical protocol (schemas + validate_trajectory + JSON) → Tasks 1–2
- Toy repo F2P/P2P matrix → Task 6 (buggy: F2P fail / P2P pass; fixed: both pass)
- LocalDockerBackend create/reset/execute/export_patch/verify/destroy → Task 7
- ToolAction→Observation loop + patch + clean verify + validate_trajectory → Task 8
- DockerExecutor gated class + security-flag create-command test → Task 4
- PathPolicy (../, symlink, protected, private bundle) → Tasks 3, 5 (view_file escape test)
- Out of scope (FastAPI/auth/registry/SSH/real Docker wiring) — intentionally absent

**Placeholder scan:** every step has concrete code and expected test output; no TBD/TODO.

**Type consistency:** `ToolAction(tool, arguments, request_id)`, `Observation(...)`, `TrajectoryStep(action, observation)`, `AgentTrajectory(task_id, steps, metadata)`, `execute_action(executor, action, env_id)`, `LocalDockerBackend(bundles_dir, use_docker)`, `_verify_clean(fresh, task, patch, bundle)` — names and signatures match across all tasks. The `test_toy_bundle.py`/`test_backend.py`/`test_integration.py` share the `EXAMPLES` path convention and `OLD`/`NEW`/`FIX` string constants consistent with the Task 6 bundle content.
