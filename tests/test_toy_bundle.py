from pathlib import Path

from sweforge.env_server.docker.backend import LocalDockerBackend, load_task_bundle
from sweforge.protocol.tools import StrReplaceAction

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
    assert result.verdict == "unresolved"
    assert result.f2p_passed == 0
    assert result.p2p_passed == 2
    assert not result.resolved


def test_fixed_f2p_and_p2p_pass():
    task = load_task_bundle(EXAMPLES).task
    backend = LocalDockerBackend(EXAMPLES.parent)
    env = backend.create(task)
    try:
        observation = backend.execute(
            env,
            StrReplaceAction(path="toy_cache/cache.py", old_string=OLD, new_string=FIX),
        )
        assert observation.success is True
        patch = backend.export_patch(env)
        result = backend.verify(task, patch)
    finally:
        backend.destroy(env)
    assert result.verdict == "resolved"
    assert result.f2p_passed == 1 and result.p2p_passed == 2
    assert result.reward == 1.0
