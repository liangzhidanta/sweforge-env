"""Schema 单元测试：TaskSpec / policy 防泄漏 / Verification / Trajectory token 层。"""

import pytest
from pydantic import ValidationError

from sweforge.schemas.task import MutationInfo, TaskEnvironment, TaskSpec, TestSpec
from sweforge.schemas.trajectory import AgentTrajectory, TerminationReason
from sweforge.schemas.verification import (
    TestResult,
    VerificationResult,
    binary_reward,
    is_resolved,
)
from tests.helpers import build_valid_trajectory

# pytest 会把导入的 TestResult/TestSpec 当成测试类收集；显式标记为非测试类
from sweforge.schemas.task import TestSpec as _TestSpec
from sweforge.schemas.verification import TestResult as _TestResult

_TestResult.__test__ = False
_TestSpec.__test__ = False


def make_task(**overrides):
    base = dict(
        task_id="task_0001",
        repo="octocat/hello",
        base_commit="abc123",
        problem_statement="Make test_foo pass.",
        environment=TaskEnvironment(
            image="python:3.12",
            setup_commands=["pip install -e ."],
            build_commands=[],
            test_commands={"test_foo": ["pytest tests/test_foo.py"]},
        ),
        fail_to_pass=[TestSpec(test_id="test_foo", kind="fail_to_pass")],
        pass_to_pass=[TestSpec(test_id="test_bar", kind="pass_to_pass")],
    )
    base.update(overrides)
    return TaskSpec(**base)


class TestTaskSpec:
    def test_policy_view_redacts_tests(self):
        task = make_task()
        policy = task.policy_view()
        assert policy.problem_statement == task.problem_statement
        assert policy.environment.test_commands == {}  # 测试命令必须剥离
        assert not hasattr(policy, "gold_patch")
        assert not hasattr(policy, "fail_to_pass")

    def test_policy_view_structurally_redacts_gold_patch(self):
        # policy_view 结构上不可能带 gold_patch（PolicyTask 只有 5 个字段）
        task = make_task(gold_patch="--- a/x\n+++ b/x")
        policy = task.policy_view()
        assert policy.model_dump_json().find("gold") == -1
        assert "--- a/x" not in policy.model_dump_json()

    def test_duplicate_test_id_rejected(self):
        with pytest.raises(ValidationError):
            make_task(
                fail_to_pass=[TestSpec(test_id="t", kind="fail_to_pass")],
                pass_to_pass=[TestSpec(test_id="t", kind="pass_to_pass")],
            )

    def test_mutation_not_in_policy_view(self):
        task = make_task(
            mutation=MutationInfo(kind="invert_condition", file="a.py", location="L42")
        )
        assert "invert_condition" not in task.policy_view().model_dump_json()


class TestVerification:
    def _resolved(self, **kw) -> VerificationResult:
        base = dict(
            verification_id="v1",
            task_id="task_0001",
            verdict="resolved",
            fail_to_pass=[TestResult(test_id="f", kind="fail_to_pass", passed=True)],
            pass_to_pass=[TestResult(test_id="p", kind="pass_to_pass", passed=True)],
            integrity_ok=True,
        )
        base.update(kw)
        return VerificationResult(**base)

    def test_is_resolved(self):
        assert is_resolved(self._resolved())
        assert not is_resolved(self._resolved(verdict="unresolved"))
        assert not is_resolved(
            self._resolved(fail_to_pass=[TestResult(test_id="f", kind="fail_to_pass", passed=False)])
        )
        assert not is_resolved(self._resolved(integrity_ok=False))

    def test_binary_reward(self):
        assert binary_reward(self._resolved()) == 1.0
        assert binary_reward(self._resolved(verdict="unresolved")) == 0.0

    def test_empty_f2p_not_resolved(self):
        # 没有 F2P 的验证不能算 resolved（无 F2P 的任务不合格）
        v = self._resolved(fail_to_pass=[], verdict="resolved")
        assert not is_resolved(v)
        # 但模型本身不阻止这种不一致 —— 由 validate_verification 捕获（见 test_validate）

    def test_verdict_error_never_resolved(self):
        v = VerificationResult(
            verification_id="v2",
            task_id="t2",
            verdict="error",
            fail_to_pass=[TestResult(test_id="f", kind="fail_to_pass", passed=True)],
            pass_to_pass=[TestResult(test_id="p", kind="pass_to_pass", passed=True)],
            integrity_ok=True,
        )
        # verdict 与测试结果的不一致由 validate_verification() 捕获
        assert is_resolved(v) is False
        assert v.resolved is False


class TestTrajectoryShapes:
    def test_mask_length_enforced(self):
        from tests.helpers import build_messages_and_turns

        messages, turns = build_messages_and_turns()
        with pytest.raises(ValidationError):
            AgentTrajectory(
                trajectory_id="bad",
                task_id="t",
                messages=messages,
                turns=turns,
                prompt_ids=[0] * 14,
                response_ids=[0] * 15,
                response_mask=[1] * 14,  # 错：应为 15
                termination_reason=TerminationReason.AGENT_FINISHED,
            )

    def test_token_count_alignment_enforced(self):
        from tests.helpers import build_messages_and_turns

        messages, turns = build_messages_and_turns()
        with pytest.raises(ValidationError):
            AgentTrajectory(
                trajectory_id="bad",
                task_id="t",
                messages=messages,
                turns=turns,
                prompt_ids=[0] * 14,
                response_ids=[0] * 15,
                response_mask=[1] * 15,
                message_token_counts=[1] * 7,  # sum=7 != 14+15
                termination_reason=TerminationReason.AGENT_FINISHED,
            )

    def test_labels_for_sft(self):
        t = build_valid_trajectory()
        labels = t.labels_for_sft(ignore_index=-100)
        # 消息序列: system(3) user(5) | a1(5) t1(3) | a2(5) t2(3) | a3(5)
        # label 与完整输入序列等长: 非 assistant 消息 -> -100，assistant 消息 -> token id
        expected = (
            [-100] * 8  # system 3 + user 5
            + t.response_ids[0:5]
            + [-100] * 3  # t1
            + t.response_ids[5:10]
            + [-100] * 3  # t2
            + t.response_ids[10:15]
        )
        assert labels == expected
        assert t.loss_token_count == 15  # 全部 assistant token 都是学习目标
        # -100 的位置恰好是全部 prompt（非 assistant）token
        assert sum(1 for x in labels if x == -100) == len(t.prompt_ids)

    def test_labels_require_token_counts(self):
        t = build_valid_trajectory()
        t.message_token_counts = None  # 模拟未记录边界的轨迹
        with pytest.raises(ValueError):
            t.labels_for_sft()
