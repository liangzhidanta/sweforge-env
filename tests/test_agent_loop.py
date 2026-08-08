"""阶段 7: Agent Loop 冒烟测试（Mock + Remote 双后端）。

脚本化 policy 跑完整 agent 循环 -> Canonical AgentTrajectory，必须通过
validate_trajectory()（§10 硬门槛: RL 轨迹与 SFT 轨迹同一校验）。
"""

from __future__ import annotations

import pytest

from sweforge.agent.loop import AgentLoop, AssistantDecision
from sweforge.environment.mock import MockEnvironmentBackend
from sweforge.environment.remote import RemoteEnvironmentBackend
from sweforge.protocol.messages import CanonicalMessage
from sweforge.protocol.serialization import render_observation
from sweforge.protocol.system_prompt import CANONICAL_SYSTEM_PROMPT
from sweforge.protocol.tools import FinishAction, StrReplaceAction, ViewFileAction
from sweforge.protocol.validate import validate_trajectory
from sweforge.schemas.task import TaskEnvironment, TaskSpec
from sweforge.schemas.trajectory import TerminationReason
from tests.conftest import SEED


class ScriptedPolicy:
    """确定性脚本 policy: 依次执行给定 action 序列（冒烟/测试用）。

    真实 policy 是阶段 10 的 veRL/vLLM rollout; 这里只验证 loop 语义。
    """

    def __init__(self, steps: list):
        self.steps = steps
        self.calls: list[list[CanonicalMessage]] = []  # 每次 decide 收到的消息历史

    def decide(self, messages):
        self.calls.append(list(messages))
        if not self.steps:
            return None
        action = self.steps.pop(0)
        return AssistantDecision(content=f"step {len(self.calls)}", action=action)


class ExplodingBackend(MockEnvironmentBackend):
    """execute 抛异常的后端（模拟环境故障）。"""

    def execute(self, env, action):
        raise RuntimeError("container exploded")


def _task(task_id: str = "demo") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        repo="demo",
        base_commit="0000000",
        problem_statement="fix the wrong answer",
        environment=TaskEnvironment(setup_commands=["true"]),
    )


def _fix_policy():
    """view -> str_replace -> finish 的标准修复脚本。"""
    return ScriptedPolicy(
        [
            ViewFileAction(path="answer.txt"),
            StrReplaceAction(path="answer.txt", old_string="wrong ", new_string=""),
            FinishAction(),
        ]
    )


def _mock_backend(tmp_path) -> MockEnvironmentBackend:
    return MockEnvironmentBackend(workspace_root=tmp_path / "envs", seed_files=SEED)


# ---------------- Mock 后端 ----------------

def test_loop_finish_produces_valid_trajectory(tmp_path):
    backend = _mock_backend(tmp_path)
    policy = _fix_policy()
    traj = AgentLoop(backend, policy).run(_task())

    assert traj.termination_reason == TerminationReason.AGENT_FINISHED
    assert traj.num_turns == 3
    v = validate_trajectory(traj)
    assert v.ok, v.errors

    # patch 导出（a/ b/ 前缀的 unified diff）
    assert traj.patch and "a/answer.txt" in traj.patch and "+the answer" in traj.patch

    # 消息结构: system -> user -> assistant -> tool -> assistant -> tool -> assistant(finish)
    roles = [m.role for m in traj.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert traj.messages[0].content == CANONICAL_SYSTEM_PROMPT
    assert traj.messages[1].content == "fix the wrong answer"
    # finish turn 无 observation（schema §2: finish 之后无 tool 消息）
    assert traj.turns[-1].observation is None
    assert traj.turns[-1].action.name == "finish"
    # tool 消息渲染 == 确定性渲染
    assert traj.messages[3].content == render_observation(traj.turns[0].observation)
    assert traj.messages[3].tool_call_id == traj.turns[0].assistant_message.tool_calls[0].id


def test_loop_max_steps_exceeded(tmp_path):
    backend = _mock_backend(tmp_path)
    policy = ScriptedPolicy([ViewFileAction(path="answer.txt")] * 5)
    traj = AgentLoop(backend, policy, max_steps=3).run(_task())
    assert traj.termination_reason == TerminationReason.MAX_STEPS_EXCEEDED
    assert traj.num_turns == 3
    assert "max_steps" in traj.metadata["error"]
    v = validate_trajectory(traj)
    assert v.ok, v.errors


def test_loop_agent_error_on_none(tmp_path):
    backend = _mock_backend(tmp_path)
    traj = AgentLoop(backend, ScriptedPolicy([])).run(_task())
    assert traj.termination_reason == TerminationReason.AGENT_ERROR
    assert traj.num_turns == 0
    v = validate_trajectory(traj)
    assert v.ok, v.errors


def test_loop_environment_error(tmp_path):
    backend = ExplodingBackend(workspace_root=tmp_path / "envs", seed_files=SEED)
    traj = AgentLoop(backend, _fix_policy()).run(_task())
    assert traj.termination_reason == TerminationReason.ENVIRONMENT_ERROR
    assert "container exploded" in traj.metadata["error"]
    # 异常前的 turn 保留（诊断价值）
    assert traj.num_turns == 1
    v = validate_trajectory(traj)
    assert v.ok, v.errors


# ---------------- Remote 后端（同一 loop 走 HTTP 契约） ----------------

def test_loop_remote_backend(tmp_path, remote_server):
    url, _ = remote_server
    backend = RemoteEnvironmentBackend(base_url=url)
    policy = _fix_policy()
    traj = AgentLoop(backend, policy).run(_task(task_id="remote-demo"))

    assert traj.termination_reason == TerminationReason.AGENT_FINISHED
    assert traj.num_turns == 3
    v = validate_trajectory(traj)
    assert v.ok, v.errors
    assert traj.patch and "a/answer.txt" in traj.patch
    assert traj.metadata["backend"] == "RemoteEnvironmentBackend"


def test_loop_remote_env_destroyed_after_run(tmp_path, remote_server):
    """loop 结束后远端环境被销毁（生命周期不外泄）。"""
    url, _ = remote_server
    backend = RemoteEnvironmentBackend(base_url=url)
    AgentLoop(backend, _fix_policy()).run(_task(task_id="remote-clean"))
    with pytest.raises(RuntimeError, match="404"):
        backend.reset("remote-clean")
