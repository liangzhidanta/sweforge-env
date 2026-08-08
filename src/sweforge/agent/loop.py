"""阶段 7: Agent Loop —— 多步 agent 循环（PROJECT_SPEC §7 / §9 / §2 消息语义）。

输入: TaskSpec + Policy（从消息历史决策）；输出: Canonical AgentTrajectory。
只依赖 EnvironmentBackend 接口（铁律 P3: 禁止 Docker 依赖）。

循环语义（与 canonical 消息协议一一对应）:
    system(CANONICAL_SYSTEM_PROMPT) -> user(task) -> [assistant(+tool_call)
    -> tool(observation 渲染) ...] -> assistant(+finish) 终止

    - 每条 assistant 消息恰好携带 1 个 tool_call（policy 每次决策都调用工具，
      finish 也是工具）; 纯文本 assistant 消息不产生（模型只会工具调用，
      注释文本放 content）——符合"0 个 tool_call 仅出现在非正常终止末尾"
    - finish 的 turn observation = None，其后无 tool 消息（schema §2）
    - 异常终止映射 TerminationReason:
        policy 返回 None / 抛异常        -> AGENT_ERROR
        backend.execute 抛异常           -> ENVIRONMENT_ERROR
        步数超限                        -> MAX_STEPS_EXCEEDED

Policy 是插件: 阶段 10 换成 veRL/vLLM rollout（SWEForgeAgentLoop），
冒烟/测试用脚本化 policy（tests/test_agent_loop.py）。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

from sweforge.environment.base import EnvironmentBackend
from sweforge.protocol.messages import CanonicalMessage, CanonicalToolCall
from sweforge.protocol.serialization import render_observation
from sweforge.protocol.system_prompt import CANONICAL_SYSTEM_PROMPT
from sweforge.protocol.tools import ToolAction
from sweforge.protocol.validate import validate_trajectory
from sweforge.schemas.task import TaskSpec
from sweforge.schemas.trajectory import AgentTrajectory, AgentTurn, TerminationReason

__all__ = ["AgentLoop", "AssistantDecision", "Policy"]


@dataclass
class AssistantDecision:
    """policy 的一次决策: assistant 注释文本 + 恰好一个工具调用。"""

    content: str | None
    action: ToolAction


class Policy(Protocol):
    """从当前消息历史决定下一条 assistant 回复（None = 无法继续）。"""

    def decide(self, messages: list[CanonicalMessage]) -> AssistantDecision | None: ...


class AgentLoop:
    """多步 agent 循环。backend 在 run 内 create/destroy（生命周期不外泄）。"""

    def __init__(self, backend: EnvironmentBackend, policy: Policy, *, max_steps: int = 40):
        self.backend = backend
        self.policy = policy
        self.max_steps = max_steps

    def run(self, task: TaskSpec, trajectory_id: str | None = None) -> AgentTrajectory:
        env = self.backend.create(task)
        messages: list[CanonicalMessage] = [
            CanonicalMessage(role="system", content=CANONICAL_SYSTEM_PROMPT),
            CanonicalMessage(role="user", content=task.problem_statement),
        ]
        turns: list[AgentTurn] = []
        termination = TerminationReason.INTERRUPTED
        error_note: str | None = None

        try:
            for _ in range(self.max_steps):
                try:
                    decision = self.policy.decide(messages)
                except Exception as e:
                    termination = TerminationReason.AGENT_ERROR
                    error_note = f"policy.decide raised: {e}"
                    break
                if decision is None:
                    termination = TerminationReason.AGENT_ERROR
                    error_note = "policy returned None (no action available)"
                    break

                tool_call = CanonicalToolCall(
                    id=uuid.uuid4().hex[:16],
                    name=decision.action.name,
                    arguments=decision.action.model_dump(exclude={"name"}),
                )
                assistant_msg = CanonicalMessage(
                    role="assistant", content=decision.content, tool_calls=[tool_call]
                )
                messages.append(assistant_msg)

                try:
                    obs = self.backend.execute(env, decision.action)
                except Exception as e:
                    # 环境故障: 记录该 turn（observation=None, 与 finish 同语义——
                    # turns 渲染必须与 messages 的 assistant/tool 部分一致）
                    turns.append(
                        AgentTurn(assistant_message=assistant_msg, action=decision.action, observation=None)
                    )
                    termination = TerminationReason.ENVIRONMENT_ERROR
                    error_note = f"backend.execute raised: {e}"
                    break

                if decision.action.name == "finish":
                    # finish 终止: turn 无 observation, 其后无 tool 消息（schema §2）
                    turns.append(
                        AgentTurn(assistant_message=assistant_msg, action=decision.action, observation=None)
                    )
                    termination = TerminationReason.AGENT_FINISHED
                    break

                turns.append(
                    AgentTurn(assistant_message=assistant_msg, action=decision.action, observation=obs)
                )
                messages.append(
                    CanonicalMessage(
                        role="tool",
                        content=render_observation(obs),
                        tool_call_id=tool_call.id,
                    )
                )
            else:
                termination = TerminationReason.MAX_STEPS_EXCEEDED
                error_note = f"max_steps ({self.max_steps}) exceeded"
        finally:
            # 无论终止方式, 都导出 patch（失败时的部分修改也有诊断价值）
            try:
                patch = self.backend.export_patch(env)
            except Exception:
                patch = None
            try:
                self.backend.destroy(env)  # 清理失败不掩盖终止语义
            except Exception:
                pass

        trajectory = AgentTrajectory(
            trajectory_id=trajectory_id or f"traj-{task.task_id}-{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
            messages=messages,
            turns=turns,
            termination_reason=termination,
            patch=patch,
            metadata={
                "agent_loop": "canonical-v1",
                "backend": getattr(self.backend, "backend_name", None)
                or type(self.backend).__name__,
                **( {"error": error_note} if error_note else {}),
            },
        )
        # §10 验收硬门槛: RL 轨迹必须与 SFT 轨迹过同一个 validate_trajectory()
        v = validate_trajectory(trajectory)
        if not v.ok:
            raise RuntimeError(
                f"agent loop produced invalid trajectory ({v.errors[0] if v.errors else 'unknown'}); "
                "fix the loop, not the validator"
            )
        return trajectory
