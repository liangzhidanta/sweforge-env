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
