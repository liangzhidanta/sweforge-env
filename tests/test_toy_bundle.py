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
    result = backend.verify(task, "")
    assert result.f2p_ratio == 0.0
    assert result.p2p_ratio == 1.0
    assert not result.resolved


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
