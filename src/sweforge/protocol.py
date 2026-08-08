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
