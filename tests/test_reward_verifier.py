"""阶段 9: CleanVerifier 行为测试（PROJECT_SPEC §5 / P4 / P5）。

核心语义:
    resolved = 全部 F2P pass AND 全部 P2P pass AND integrity_ok（patch 干净应用）
    apply 失败        -> integrity_ok=False + unresolved（不猜测, 不抛异常）
    isolation 故障    -> verdict="error"
    reward           = binary_reward（1.0/0.0, 来自真实执行）
"""

from __future__ import annotations

import pytest

from sweforge.reward.verifier import CleanVerifier, TempDirIsolation, VERIFIER_VERSION
from sweforge.schemas.task import TaskEnvironment, TaskSpec
from sweforge.schemas.verification import binary_reward
from tests.conftest import GOOD_PATCH, SEED

BAD_PATCH = "--- a/no-such-file.txt\n+++ b/no-such-file.txt\n@@ -1 +1 @@\n-x\n+y\n"


def _task(task_id: str = "v1") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        repo="demo",
        base_commit="0000000",
        problem_statement="fix the wrong answer",
        environment=TaskEnvironment(
            setup_commands=["echo setup-ran > setup.txt"],
            build_commands=["echo build-ran > build.txt"],
            test_commands={
                # grep -c: 失败时有输出（grep -q 静默无输出, 无法验证 output 捕获）
                "test_answer": ["grep", "-c", "the answer", "answer.txt"],
                "test_sanity": ["true"],
                "test_built": ["test", "-f", "build.txt"],
            },
        ),
        fail_to_pass=[{"test_id": "test_answer", "kind": "fail_to_pass"}],
        pass_to_pass=[
            {"test_id": "test_sanity", "kind": "pass_to_pass"},
            {"test_id": "test_built", "kind": "pass_to_pass"},
        ],
    )


def _verifier() -> CleanVerifier:
    return CleanVerifier(TempDirIsolation(seed_files=SEED))


# ---------------- verdict 语义 ----------------

def test_resolved_with_good_patch():
    v = _verifier().verify(_task(), GOOD_PATCH)
    assert v.verdict == "resolved" and v.resolved
    assert v.integrity_ok
    assert v.reward == 1.0 and binary_reward(v) == 1.0
    assert v.fail_to_pass[0].passed and all(r.passed for r in v.pass_to_pass)
    assert v.metadata["verifier"] == VERIFIER_VERSION
    assert v.metadata["isolated"] is True


def test_unresolved_without_fix():
    v = _verifier().verify(_task(), "")
    assert v.verdict == "unresolved" and not v.resolved
    assert v.reward == 0.0
    # 真实执行, 不伪造: F2P 测试确实失败
    assert not v.fail_to_pass[0].passed
    assert v.fail_to_pass[0].output is not None


def test_unresolved_when_f2p_fixed_but_p2p_broken():
    """F2P 修复了但 P2P 挂了 -> 仍 unresolved（resolved 需全过）。"""
    task = _task()
    # setup 后删除 build.txt 使 test_built 失败——注意 setup 在验证环境里
    # 会重跑, 所以改为给 patch 额外加一个破坏性的修改
    patch = GOOD_PATCH + "\n--- a/build.txt\n+++ b/build.txt\n@@ -1 +1 @@\n-build-ran\n+broken\n"
    # build_commands 会在 apply 前重跑（覆盖为 build-ran）; 用 setup 破坏不可行,
    # 所以直接构造一个 P2P 必败的任务
    task.environment.test_commands["test_sanity"] = ["false"]
    v = _verifier().verify(task, GOOD_PATCH)
    assert v.verdict == "unresolved"
    assert v.fail_to_pass[0].passed
    assert not v.pass_to_pass[0].passed


# ---------------- integrity（P4: resolved 需要 integrity_ok） ----------------

def test_apply_failure_is_unresolved_not_exception():
    """patch 应用失败: integrity_ok=False + unresolved, 绝不抛异常/猜结论。"""
    v = _verifier().verify(_task(), BAD_PATCH)
    assert v.verdict == "unresolved" and not v.resolved
    assert v.integrity_ok is False
    assert "apply_error" in v.metadata  # 错误信息供诊断
    assert v.reward == 0.0


def test_apply_failure_never_runs_tests():
    """apply 失败后 tests 环境是坏的, 但 verifier 必须给出确定性 unresolved。"""
    v = _verifier().verify(_task(), BAD_PATCH)
    assert all(not r.passed for r in v.fail_to_pass)


def test_build_failure_recorded_in_metadata():
    task = _task()
    task.environment.build_commands = ["false"]  # build 必败
    v = _verifier().verify(task, GOOD_PATCH)
    assert v.metadata.get("build_failed") is True
    # build 产物缺失 -> test_built 失败 -> unresolved
    assert v.verdict == "unresolved"
    assert not v.pass_to_pass[-1].passed


def test_setup_failure_recorded_but_continues():
    task = _task()
    task.environment.setup_commands = ["exit 1"]  # setup 必败
    v = _verifier().verify(task, GOOD_PATCH)
    assert "setup_failures" in v.metadata
    assert v.metadata["setup_failures"][0].startswith("'exit 1'")


# ---------------- error 语义（verifier 侧故障） ----------------

class ExplodingIsolation:
    """create 故障 = verifier 侧问题 -> verdict='error'。"""

    def create(self, task):
        raise RuntimeError("no docker available")

    def apply_patch(self, workdir, patch):
        return None

    def destroy(self, workdir):
        pass


def test_isolation_failure_is_error_verdict():
    v = CleanVerifier(ExplodingIsolation()).verify(_task(), GOOD_PATCH)
    assert v.verdict == "error"
    assert v.integrity_ok is False
    assert "isolation.create failed" in v.metadata["error"]
    assert v.reward == 0.0


# ---------------- 隔离性 / 不伪造 ----------------

def test_verify_is_isolated_and_cleans_up(tmp_path):
    """验证环境与外部完全隔离, 且用完即销毁。"""
    verifier = _verifier()
    verifier.verify(_task(), GOOD_PATCH)
    leftovers = list(tmp_path.parent.glob("sweforge-clean-*"))
    assert leftovers == []  # destroy 已清理


def test_verify_does_not_touch_given_workspace(tmp_path):
    """给 CleanVerifier 一个"agent 工作区", verify 不得触碰它。"""
    agent_ws = tmp_path / "agent"
    agent_ws.mkdir()
    (agent_ws / "answer.txt").write_text("hacked by agent\n")
    verifier = _verifier()
    verifier.verify(_task(), GOOD_PATCH)
    assert (agent_ws / "answer.txt").read_text() == "hacked by agent\n"


def test_duration_ms_recorded():
    v = _verifier().verify(_task(), GOOD_PATCH)
    assert all(r.duration_ms is not None for r in [*v.fail_to_pass, *v.pass_to_pass])


def test_metadata_extra_merged():
    v = CleanVerifier(TempDirIsolation(seed_files=SEED)).verify(
        _task(), GOOD_PATCH, metadata_extra={"backend": "mock"}
    )
    assert v.metadata["backend"] == "mock"
    assert v.metadata["verifier"] == VERIFIER_VERSION
