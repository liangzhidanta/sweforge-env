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


def test_verify_respects_bundle_integrity_protected(tmp_path):
    import json
    import shutil

    bundle_dir = tmp_path / "toy_cache_aliasing"
    shutil.copytree(EXAMPLES, bundle_dir)
    manifest_path = bundle_dir / "task_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity_protected"] = ["secret.txt"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (bundle_dir / "repo" / "secret.txt").write_text("classified", encoding="utf-8")

    task = load_task_bundle(bundle_dir).task
    backend = LocalDockerBackend(tmp_path)
    patch = "diff --git a/secret.txt b/secret.txt\n--- a/secret.txt\n+++ b/secret.txt\n@@ -1 +1 @@\n-classified\n+owned\n"
    result = backend.verify(task, patch)
    assert not result.integrity_ok
    assert "secret.txt" in result.details["integrity"]


def test_verify_apply_failure_reports_reason():
    backend = LocalDockerBackend(EXAMPLES.parent)
    malformed = ("diff --git a/toy_cache/cache.py b/toy_cache/cache.py\n"
                 "--- a/toy_cache/cache.py\n"
                 "+++ b/toy_cache/cache.py\n"
                 "@@ not a hunk header @@\n")
    result = backend.verify(_task(), malformed)
    assert not result.integrity_ok
    assert "git apply failed" in result.details["integrity"]


def test_verify_hidden_test_path_collision_collapses_to_integrity_failure():
    backend = LocalDockerBackend(EXAMPLES.parent)
    collision = ("diff --git a/tests/test_f2p.py/evil.txt b/tests/test_f2p.py/evil.txt\n"
                 "new file mode 100644\n"
                 "--- /dev/null\n"
                 "+++ b/tests/test_f2p.py/evil.txt\n"
                 "@@ -0,0 +1 @@\n"
                 "+owned\n")
    result = backend.verify(_task(), collision)
    assert not result.integrity_ok
    assert "hidden tests" in result.details["integrity"]


def test_verify_uses_manifest_protected_paths_not_caller(tmp_path):
    import json
    import shutil
    from dataclasses import replace

    bundle_dir = tmp_path / "toy_cache_aliasing"
    shutil.copytree(EXAMPLES, bundle_dir)
    manifest_path = bundle_dir / "task_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task"]["protected_paths"] = [".git", "secret.txt"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (bundle_dir / "repo" / "secret.txt").write_text("classified", encoding="utf-8")

    task = load_task_bundle(bundle_dir).task
    backend = LocalDockerBackend(tmp_path)
    spoofed = replace(task, protected_paths=())
    patch = ("diff --git a/secret.txt b/secret.txt\n"
             "--- a/secret.txt\n+++ b/secret.txt\n"
             "@@ -1 +1 @@\n"
             "-classified\n"
             "+owned\n")
    result = backend.verify(spoofed, patch)
    assert not result.integrity_ok
    assert "secret.txt" in result.details["integrity"]
