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
