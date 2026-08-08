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
