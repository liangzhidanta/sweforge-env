"""validate_trajectory / validate_task_spec / validate_verification 测试。

重点: SFT 轨迹与 RL 轨迹必须通过同一个 validate_trajectory()（§10 验收硬门槛）。
"""

import pytest

from sweforge.protocol.messages import CanonicalMessage, CanonicalToolCall
from sweforge.protocol.validate import (
    validate_task_spec,
    validate_trajectory,
    validate_verification,
)
from sweforge.schemas.trajectory import TerminationReason
from sweforge.schemas.verification import TestResult, VerificationResult

# pytest 会把导入的 TestResult 当成测试类收集；显式标记为非测试类
TestResult.__test__ = False
from tests.helpers import build_valid_trajectory


def make_verification(*, verdict="resolved", f2p_pass=True, p2p_pass=True, integrity=True):
    return VerificationResult(
        verification_id="verif_0001",
        task_id="task_0001",
        patch=build_valid_trajectory().patch,
        verdict=verdict,
        fail_to_pass=[TestResult(test_id="f1", kind="fail_to_pass", passed=f2p_pass)],
        pass_to_pass=[TestResult(test_id="p1", kind="pass_to_pass", passed=p2p_pass)],
        integrity_ok=integrity,
        reward=1.0 if (verdict == "resolved" and f2p_pass and p2p_pass and integrity) else 0.0,
    )


class TestValidateTrajectoryHappyPath:
    def test_sft_trajectory_passes(self):
        t = build_valid_trajectory()
        r = validate_trajectory(t)
        assert r.ok, r.errors

    def test_sft_trajectory_with_patch_passes_strict(self):
        t = build_valid_trajectory()
        r = validate_trajectory(t, require_patch_on_finish=True)
        assert r.ok, r.errors

    def test_rl_trajectory_passes_same_validator(self):
        """§10 验收: RL 轨迹与 SFT 轨迹过同一个 validate_trajectory()。"""
        t = build_valid_trajectory(trajectory_id="traj_rl_0001")
        t.reward = 1.0
        t.verification = make_verification()
        t.logprobs = [-0.5] * 15
        r = validate_trajectory(t, require_patch_on_finish=True, binary_reward_expected=True)
        assert r.ok, r.errors

    def test_rl_unresolved_passes_with_zero_reward(self):
        t = build_valid_trajectory(trajectory_id="traj_rl_0002")
        t.reward = 0.0
        t.verification = make_verification(verdict="unresolved", f2p_pass=False)
        r = validate_trajectory(t, binary_reward_expected=True)
        assert r.ok, r.errors

    def test_abnormal_termination_without_finish(self):
        """超时终止: 最后一轮没有 tool observation，也允许通过。"""
        t = build_valid_trajectory(trajectory_id="traj_timeout")
        t.messages = t.messages[:4]  # system, user, a1(bash), t1
        t.turns = t.turns[:1]
        t.termination_reason = TerminationReason.TIME_LIMIT_EXCEEDED
        t.patch = None
        t.prompt_ids = t.prompt_ids[:11]  # system3 + user5 + t1(3) = 11
        t.response_ids = t.response_ids[:5]
        t.response_mask = [1] * 5
        t.message_token_counts = [3, 5, 5, 3]
        r = validate_trajectory(t)
        assert r.ok, r.errors


class TestValidateTrajectoryFailures:
    def test_bad_message_order(self):
        t = build_valid_trajectory()
        t.messages = list(reversed(t.messages))
        r = validate_trajectory(t)
        assert not r.ok
        assert any("messages[0]" in e for e in r.errors)

    def test_tool_not_following_assistant(self):
        t = build_valid_trajectory()
        # 把第二个 tool 消息提到第一个 tool 消息前面
        m = list(t.messages)
        m[3], m[4] = m[4], m[3]
        t.messages = m
        r = validate_trajectory(t)
        assert not r.ok
        assert any("must follow an assistant" in e for e in r.errors)

    def test_termination_inconsistent(self):
        t = build_valid_trajectory()
        t.termination_reason = TerminationReason.MAX_STEPS_EXCEEDED
        r = validate_trajectory(t)
        assert not r.ok
        assert any("last message is finish" in e for e in r.errors)

    def test_finished_without_patch_strict(self):
        t = build_valid_trajectory(include_patch=False)
        r = validate_trajectory(t, require_patch_on_finish=True)
        assert not r.ok
        assert any("requires a patch" in e for e in r.errors)

    def test_message_render_mismatch(self):
        t = build_valid_trajectory()
        t.messages[3] = CanonicalMessage(role="tool", content="fabricated output", tool_call_id="call_1")
        r = validate_trajectory(t)
        assert not r.ok
        assert any("!= render of turns" in e for e in r.errors)

    def test_action_name_mismatch(self):
        from sweforge.protocol.tools import SearchAction

        t = build_valid_trajectory()
        t.turns[0].action = SearchAction(query="find bug")  # action 换成 search 但 call 还是 bash
        r = validate_trajectory(t)
        assert not r.ok
        assert any("action name" in e for e in r.errors)

    def test_token_count_mismatch(self):
        t = build_valid_trajectory()
        t.message_token_counts[0] = 99
        r = validate_trajectory(t)
        assert not r.ok
        assert any("token count mismatch" in e for e in r.errors)

    def test_reward_mismatch_verification(self):
        t = build_valid_trajectory(trajectory_id="traj_rl_bad")
        t.reward = 0.0
        t.verification = make_verification()  # resolved -> binary reward 1.0
        r = validate_trajectory(t, binary_reward_expected=True)
        assert not r.ok
        assert any("reward" in e for e in r.errors)

    def test_broken_tool_call_chain(self):
        t = build_valid_trajectory()
        t.messages[3] = CanonicalMessage(role="tool", content="x", tool_call_id="ghost")
        r = validate_trajectory(t)
        assert not r.ok
        assert any("unknown tool_call_id" in e for e in r.errors)

    def test_turn_without_call_in_middle(self):
        t = build_valid_trajectory()
        t.turns[1].assistant_message = CanonicalMessage(
            role="assistant", content="no call here"
        )
        r = validate_trajectory(t)
        assert not r.ok
        assert any("no tool_call" in e for e in r.errors)


class TestValidateTaskSpec:
    def test_valid_task(self):
        from tests.test_schemas import make_task

        r = validate_task_spec(make_task())
        assert r.ok, r.errors

    def test_missing_f2p(self):
        from tests.test_schemas import make_task

        r = validate_task_spec(make_task(fail_to_pass=[]))
        assert not r.ok
        assert any("fail_to_pass" in e for e in r.errors)

    def test_secret_leak_detected(self):
        from tests.test_schemas import make_task

        # 任务池卫生: gold_patch 未剥离 -> 拒绝
        r = validate_task_spec(make_task(gold_patch="--- a/x"))
        assert not r.ok
        assert any("gold_patch" in e for e in r.errors)


class TestValidateVerification:
    def test_consistent(self):
        r = validate_verification(make_verification())
        assert r.ok, r.errors

    def test_verdict_inconsistent(self):
        v = make_verification(verdict="unresolved")  # 测试全过但标 unresolved
        r = validate_verification(v)
        assert not r.ok

    def test_error_verdict_with_all_passing(self):
        v = make_verification(verdict="error")
        r = validate_verification(v)
        assert not r.ok
