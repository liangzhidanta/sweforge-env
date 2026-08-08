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
