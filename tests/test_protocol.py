import json

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


def test_str_replace_missing_single_argument_flagged():
    trajectory = AgentTrajectory("t1", steps=(_step("str_replace", {"path": "a.py", "old": "x"}),))
    errors = validate_trajectory(trajectory)
    assert any("missing arguments ['new']" in error for error in errors)


def test_search_missing_pattern_flagged():
    trajectory = AgentTrajectory("t1", steps=(_step("search", {"max_results": 5}),))
    errors = validate_trajectory(trajectory)
    assert any("missing arguments ['pattern']" in error for error in errors)


def test_json_round_trip():
    action = ToolAction("bash", {"command": "pwd"}, request_id="r9")
    observation = Observation(request_id="r9", env_id="e1", tool="bash", exit_code=0,
                              stdout="out", stderr="err", content="c", truncated=True, duration_ms=12)
    trajectory = AgentTrajectory("t1", steps=(TrajectoryStep(action, observation),))
    parsed = trajectory_from_dict(json.loads(trajectory_to_json(trajectory)))
    assert parsed == trajectory
