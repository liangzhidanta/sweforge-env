"""测试共用 fixture：构造合法/非法的 Canonical AgentTrajectory。"""

from __future__ import annotations

from sweforge.protocol.messages import CanonicalMessage, CanonicalToolCall
from sweforge.protocol.tools import (
    BashAction,
    BashObservation,
    FinishAction,
    SearchAction,
    SearchObservation,
    StrReplaceAction,
    StrReplaceObservation,
)
from sweforge.schemas.trajectory import AgentTrajectory, AgentTurn, TerminationReason

SYSTEM_PROMPT = "You are a coding agent. Tools: bash, search, view_file, str_replace, finish."
TASK_STATEMENT = "Fix the bug in module_a.py so that test_foo passes."


def build_messages_and_turns():
    """一条合法的 3-turn 轨迹（bash -> str_replace -> finish）。

    Returns:
        (messages, turns) —— 与 render_turns_to_messages 的输出一致。
    """
    m_system = CanonicalMessage(role="system", content=SYSTEM_PROMPT)
    m_user = CanonicalMessage(role="user", content=TASK_STATEMENT)

    # turn 1: bash
    a1 = BashAction(command="pytest tests/test_foo.py")
    o1 = BashObservation(exit_code=1, stdout="FAIL test_foo\nassert x == 42", stderr="")
    m1 = CanonicalMessage(
        role="assistant",
        content="Let me run the test.",
        tool_calls=[CanonicalToolCall(id="call_1", name="bash", arguments={"command": a1.command})],
    )
    t1 = CanonicalMessage(role="tool", content="[exit code 1]\nFAIL test_foo\nassert x == 42", tool_call_id="call_1")
    turn1 = AgentTurn(assistant_message=m1, action=a1, observation=o1)

    # turn 2: str_replace
    a2 = StrReplaceAction(path="module_a.py", old_string="x = 41", new_string="x = 42")
    o2 = StrReplaceObservation(success=True, path="module_a.py")
    m2 = CanonicalMessage(
        role="assistant",
        content="The value is off by one.",
        tool_calls=[CanonicalToolCall(id="call_2", name="str_replace", arguments={"path": "module_a.py", "old_string": "x = 41", "new_string": "x = 42"})],
    )
    t2 = CanonicalMessage(role="tool", content="[str_replace applied] module_a.py", tool_call_id="call_2")
    turn2 = AgentTurn(assistant_message=m2, action=a2, observation=o2)

    # turn 3: finish
    a3 = FinishAction(summary="Fixed the off-by-one.")
    m3 = CanonicalMessage(
        role="assistant",
        content="Done.",
        tool_calls=[CanonicalToolCall(id="call_3", name="finish", arguments={"summary": "Fixed the off-by-one."})],
    )
    turn3 = AgentTurn(assistant_message=m3, action=a3, observation=None)

    messages = [m_system, m_user, m1, t1, m2, t2, m3]
    turns = [turn1, turn2, turn3]
    return messages, turns


def build_valid_trajectory(
    *,
    trajectory_id: str = "traj_sft_0001",
    task_id: str = "task_0001",
    include_patch: bool = True,
    message_token_counts: list[int] | None = None,
) -> AgentTrajectory:
    """默认构造一条通过 validate_trajectory() 的 SFT 轨迹。

    message_token_counts 默认按 messages 顺序给定（system 3, user 5, 每 assistant
    5, 每 tool 3；与 prompt/response ids 对齐）。
    """
    messages, turns = build_messages_and_turns()
    if message_token_counts is None:
        message_token_counts = [3, 5, 5, 3, 5, 3, 5]  # = messages 长度
    # prompt 区: system(3) + user(5) + tool1(3) + tool2(3) = 14
    # response 区: assistant 5+5+5 = 15
    prompt_ids = list(range(100, 114))
    response_ids = list(range(200, 215))
    return AgentTrajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        messages=messages,
        turns=turns,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=[1] * 15,
        message_token_counts=message_token_counts,
        termination_reason=TerminationReason.AGENT_FINISHED,
        patch="--- a/module_a.py\n+++ b/module_a.py\n@@ -1 +1 @@\n-x = 41\n+x = 42\n" if include_patch else None,
        metadata={
            "protocol_version": "canonical-v1",
            "source": "r2e-gym",
            "source_record_id": "r2e-gym-42",
            "source_version": "r2e-gym-v1",
            "license": "MIT",
            "success_field": "is_solved",
        },
    )
