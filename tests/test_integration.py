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
