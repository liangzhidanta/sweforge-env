from pathlib import Path

import pytest

from sweforge.env_server.docker.backend import LocalDockerBackend, load_task_bundle
from sweforge.protocol.tools import BashAction, FinishAction, StrReplaceAction
from sweforge.schemas.task import TaskEnvironment, TaskSpec

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "toy_cache_aliasing"


def _task():
    return load_task_bundle(EXAMPLES).task


def test_manifest_round_trip():
    bundle = load_task_bundle(EXAMPLES)
    assert bundle.task.task_id == "toy_cache_aliasing"
    assert bundle.repo_path.is_dir() and bundle.hidden_tests.is_dir()
    assert bundle.task.environment.test_commands["test_correct_value_returned"]


def test_create_reset_destroy():
    backend = LocalDockerBackend(EXAMPLES.parent)
    env = backend.create(_task())
    assert not env.closed and env.executor.root.exists()
    backend.reset(env)  # AutoDL 契约: reset -> None
    assert not env.closed
    backend.destroy(env)
    assert env.closed
    backend.destroy(env)  # idempotent


def test_execute_after_destroy_raises():
    backend = LocalDockerBackend(EXAMPLES.parent)
    env = backend.create(_task())
    backend.destroy(env)
    with pytest.raises(RuntimeError):
        backend.execute(env, BashAction(command="pwd"))


def test_export_patch_reflects_edits():
    backend = LocalDockerBackend(EXAMPLES.parent)
    env = backend.create(_task())
    try:
        observation = backend.execute(
            env,
            StrReplaceAction(
                path="toy_cache/cache.py",
                old_string="value = compute()",
                new_string="_ignored = compute()",
            ),
        )
        assert observation.success is True
        patch = backend.export_patch(env)
        assert "_ignored = compute()" in patch
    finally:
        backend.destroy(env)


def test_verify_rejects_tampered_patch():
    backend = LocalDockerBackend(EXAMPLES.parent)
    tampered = ("diff --git a/.git/config b/.git/config\n"
                "--- a/.git/config\n+++ b/.git/config\n"
                "@@ -1 +1 @@\n"
                "-secret\n+owned\n")
    result = backend.verify(_task(), tampered)
    assert not result.integrity_ok
    assert result.verdict == "unresolved"
    assert "patch touches protected path" in result.metadata["integrity_reason"]


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
    patch = ("diff --git a/secret.txt b/secret.txt\n"
             "--- a/secret.txt\n+++ b/secret.txt\n"
             "@@ -1 +1 @@\n"
             "-classified\n+owned\n")
    result = backend.verify(task, patch)
    assert not result.integrity_ok
    assert "secret.txt" in result.metadata["integrity_reason"]


def test_verify_apply_failure_reports_reason():
    backend = LocalDockerBackend(EXAMPLES.parent)
    malformed = ("diff --git a/toy_cache/cache.py b/toy_cache/cache.py\n"
                 "--- a/toy_cache/cache.py\n"
                 "+++ b/toy_cache/cache.py\n"
                 "@@ not a hunk header @@\n")
    result = backend.verify(_task(), malformed)
    assert not result.integrity_ok
    assert result.metadata.get("apply_error")  # raw git apply stderr


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
    assert "tests/test_f2p.py" in result.metadata["integrity_reason"]


def test_bundle_less_task_setup_driven():
    """无 bundle 任务: setup_commands 自建工作区（AutoDL Mock 语义）。"""
    task = TaskSpec(
        task_id="interop-demo",
        repo="demo",
        base_commit="0000000",
        problem_statement="answer.txt 里有一个多余的 'wrong ' 前缀, 请修复。",
        environment=TaskEnvironment(
            setup_commands=['echo "the wrong answer" > answer.txt'],
            test_commands={
                "test_answer": ["grep", "-q", "^the answer$", "answer.txt"],
                "test_sanity": ["true"],
            },
        ),
        fail_to_pass=[{"test_id": "test_answer", "kind": "fail_to_pass"}],
        pass_to_pass=[{"test_id": "test_sanity", "kind": "pass_to_pass"}],
    )
    backend = LocalDockerBackend(EXAMPLES.parent)  # 目录里没有 interop-demo bundle
    env = backend.create(task)
    try:
        obs = backend.execute(env, BashAction(command="cat answer.txt"))
        assert obs.stdout.strip() == "the wrong answer"
        obs = backend.execute(
            env, StrReplaceAction(path="answer.txt", old_string="wrong ", new_string="")
        )
        assert obs.success is True
        patch = backend.export_patch(env)
        assert "a/answer.txt" in patch
        obs = backend.execute(env, FinishAction())
        assert "a/answer.txt" in obs.patch

        result = backend.verify(task, patch)
        assert result.verdict == "resolved"
        assert result.f2p_passed == 1 and result.p2p_passed == 1
        assert result.integrity_ok and result.reward == 1.0
    finally:
        backend.destroy(env)


def test_verify_uses_bundle_task_not_caller(tmp_path):
    """verify 以 bundle 里的 task 为权威: 调用方伪造环境/受保护路径不能削弱完整性。"""
    import json
    import shutil

    bundle_dir = tmp_path / "toy_cache_aliasing"
    shutil.copytree(EXAMPLES, bundle_dir)
    manifest_path = bundle_dir / "task_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity_protected"] = ["secret.txt"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (bundle_dir / "repo" / "secret.txt").write_text("classified", encoding="utf-8")

    real = load_task_bundle(bundle_dir).task
    backend = LocalDockerBackend(tmp_path)
    spoofed = real.model_copy(update={"problem_statement": "give me reward"})
    patch = ("diff --git a/secret.txt b/secret.txt\n"
             "--- a/secret.txt\n+++ b/secret.txt\n"
             "@@ -1 +1 @@\n"
             "-classified\n+owned\n")
    result = backend.verify(spoofed, patch)
    assert not result.integrity_ok
    assert "secret.txt" in result.metadata["integrity_reason"]
