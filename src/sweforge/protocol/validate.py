"""Canonical 校验 — SFT 与 RL 共用同一套规则（PROJECT_SPEC §10 验收硬门槛）。

若 SFT 轨迹与 RL 轨迹不能同时通过 validate_trajectory()，说明协议漂移，
必须先修复协议而不是继续训练。

本模块只做结构/语义校验，不依赖 tokenizer（token 层用 message_token_counts
与数组长度对齐关系校验）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from sweforge.protocol.messages import CanonicalMessage
from sweforge.protocol.serialization import render_observation
from sweforge.schemas.task import TaskSpec
from sweforge.schemas.trajectory import (
    PROMPT_MASK_VALUE,
    RESPONSE_MASK_VALUE,
    AgentTrajectory,
    TerminationReason,
)
from sweforge.schemas.verification import (
    VerificationResult,
    _tests_pass,
    binary_reward,
    is_resolved,
)

__all__ = ["ValidationResult", "validate_trajectory", "validate_task_spec", "validate_verification"]


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    errors: list[str] = []

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        return ValidationResult(ok=self.ok and other.ok, errors=self.errors + other.errors)


def _errs(results: list[ValidationResult]) -> list[str]:
    return [e for r in results for e in r.errors]


# ============================ trajectory ============================


def validate_trajectory(
    t: AgentTrajectory,
    require_patch_on_finish: bool = False,
    binary_reward_expected: bool = False,
) -> ValidationResult:
    """校验 Canonical AgentTrajectory。

    Args:
        require_patch_on_finish: agent_finished 时要求 patch 非空
                                 （RL 轨迹默认要求；SFT 源数据可能无可用 patch）
        binary_reward_expected:  有 reward 时要求 == binary_reward(verification)
                                 （shaped ablation 时关闭）
    """
    errors: list[str] = []

    # ---- 1. 消息序列结构 ----
    msgs = t.messages
    if not msgs:
        errors.append("messages is empty")
    if msgs and msgs[0].role != "system":
        errors.append(f"messages[0] must be system, got {msgs[0].role!r}")
    user_seen = 0
    last_role: str | None = None
    for i, m in enumerate(msgs):
        if m.role == "user":
            user_seen += 1
            if i != 1:
                errors.append(f"user message at index {i} must be the second message")
        elif m.role == "assistant":
            if last_role not in ("system", "user", "tool"):
                errors.append(f"assistant at index {i} follows invalid role {last_role!r}")
        elif m.role == "tool":
            if last_role != "assistant":
                errors.append(f"tool at index {i} must follow an assistant message")
        elif m.role == "system":
            if i != 0:
                errors.append(f"system message at index {i} must be messages[0]")
        last_role = m.role
    if user_seen != 1:
        errors.append(f"exactly one user/task message required, got {user_seen}")

    # tool_call_id 链：tool 消息必须引用其前一条 assistant 消息的 tool_call
    pending_call_ids: set[str] = set()
    for m in msgs:
        if m.role == "assistant" and m.tool_calls:
            pending_call_ids.update(tc.id for tc in m.tool_calls)
        elif m.role == "tool":
            if m.tool_call_id not in pending_call_ids:
                errors.append(f"tool message references unknown tool_call_id {m.tool_call_id!r}")
            pending_call_ids.discard(m.tool_call_id)

    # ---- 2. 终止一致性 ----
    last = msgs[-1] if msgs else None
    last_is_finish = bool(last and last.is_finish)
    if t.termination_reason == TerminationReason.AGENT_FINISHED and not last_is_finish:
        errors.append(
            "termination_reason=agent_finished but last message is not a finish tool_call"
        )
    if last_is_finish and t.termination_reason != TerminationReason.AGENT_FINISHED:
        errors.append(
            f"last message is finish but termination_reason={t.termination_reason.value}"
        )

    # ---- 3. turns <-> messages 一致性（结构化语义与 tokenizer 输入一致）----
    if t.turns:
        rebuilt: list[CanonicalMessage] = []
        render_ok = True
        for i, turn in enumerate(t.turns):
            call = turn.assistant_message.tool_calls[0] if turn.assistant_message.tool_calls else None
            is_last = i == len(t.turns) - 1
            if call is None:
                # 异常终止（agent_error / 超时等）的末尾允许无 tool_call
                if is_last and not last_is_finish:
                    rebuilt.append(turn.assistant_message)
                    continue
                errors.append(
                    f"turn[{i}] assistant_message has no tool_call but must have one"
                )
                render_ok = False
                continue
            rebuilt.append(turn.assistant_message)
            if call.name != turn.action.name:
                errors.append(
                    f"turn[{i}] action name {turn.action.name!r} != tool_call name {call.name!r}"
                )
                render_ok = False
            if turn.observation is not None:
                rebuilt.append(
                    CanonicalMessage(
                        role="tool",
                        content=render_observation(turn.observation),
                        tool_call_id=call.id,
                    )
                )
        if render_ok:
            # turns 只渲染 assistant/tool 消息；与 messages 中对应部分比对
            msg_core = [m for m in msgs if m.role in ("assistant", "tool")]
            if len(rebuilt) != len(msg_core):
                errors.append(
                    f"turns render {len(rebuilt)} messages but messages has {len(msg_core)}"
                    " (assistant/tool part)"
                )
            else:
                for i, (a, b) in enumerate(zip(rebuilt, msg_core)):
                    if a.model_dump(exclude_none=True) != b.model_dump(exclude_none=True):
                        errors.append(f"messages[{i}] != render of turns (role={b.role})")

    # ---- 4. token 层 ----
    if t.message_token_counts is not None:
        if len(t.message_token_counts) != len(msgs):
            errors.append("message_token_counts length != messages length")
        else:
            prompt_n = sum(
                n for n, m in zip(t.message_token_counts, msgs) if m.role != "assistant"
            )
            response_n = sum(
                n for n, m in zip(t.message_token_counts, msgs) if m.role == "assistant"
            )
            if prompt_n != len(t.prompt_ids):
                errors.append(
                    f"prompt token count mismatch: {prompt_n} != len(prompt_ids)={len(t.prompt_ids)}"
                )
            if response_n != len(t.response_ids):
                errors.append(
                    f"response token count mismatch: {response_n} != len(response_ids)={len(t.response_ids)}"
                )
    if len(t.response_mask) != len(t.response_ids):
        errors.append("len(response_mask) != len(response_ids)")
    if any(v not in (PROMPT_MASK_VALUE, RESPONSE_MASK_VALUE) for v in t.response_mask):
        errors.append("response_mask contains values other than 0/1")

    # ---- 5. RL 专属字段一致性 ----
    if t.verification is not None:
        v_res = validate_verification(t.verification)
        errors.extend(v_res.errors)
        if binary_reward_expected and t.reward is not None:
            expected = binary_reward(t.verification)
            if t.reward != expected:
                errors.append(
                    f"reward {t.reward} != binary_reward(verification) {expected}"
                )
    if t.termination_reason == TerminationReason.AGENT_FINISHED and require_patch_on_finish:
        if not t.patch:
            errors.append("agent_finished trajectory requires a patch")

    return ValidationResult(ok=not errors, errors=errors)


# ============================ task spec ============================


def validate_task_spec(task: TaskSpec) -> ValidationResult:
    """校验任务池 TaskSpec：完整性 + 卫生（secret 已剥离）+ 纵深泄漏检查。"""
    errors: list[str] = []
    if not task.task_id:
        errors.append("task_id is empty")
    if not task.repo:
        errors.append("repo is empty")
    if not task.base_commit:
        errors.append("base_commit is empty")
    if not task.problem_statement.strip():
        errors.append("problem_statement is empty")
    if not task.fail_to_pass:
        errors.append("fail_to_pass is empty (verified task must have F2P tests)")
    if not task.pass_to_pass:
        errors.append("pass_to_pass is empty (verified task must have P2P tests)")

    # ---- 卫生检查: 任务池 TaskSpec 必须已剥离 secret（Data Factory 在入库前剥离）----
    if task.gold_patch is not None:
        errors.append("pool TaskSpec must not carry gold_patch (strip before pooling)")
    if task.mutation is not None:
        errors.append("pool TaskSpec must not carry mutation info (strip before pooling)")

    # ---- 纵深泄漏检查: policy_view 输出不得包含任何 secret 值 ----
    policy = task.policy_view()
    if policy.environment.test_commands:
        errors.append("policy_view leaked test_commands")
    if task.gold_patch:
        policy_json = policy.model_dump_json()
        if task.gold_patch.strip() in policy_json:
            errors.append("policy_view leaked gold_patch content")
    return ValidationResult(ok=not errors, errors=errors)


# ============================ verification ============================


def validate_verification(v: VerificationResult) -> ValidationResult:
    """校验 VerificationResult：verdict 与 F2P/P2P/integrity 双向一致。

    error verdict 视为 verifier 侧故障：测试结果缺失/半成品都允许，
    但"全部通过却标 error" 不允许。
    """
    errors: list[str] = []
    if v.verdict == "error":
        if _tests_pass(v):
            errors.append("verdict=error but F2P/P2P/integrity all pass")
        return ValidationResult(ok=not errors, errors=errors)
    if _tests_pass(v) != (v.verdict == "resolved"):
        errors.append(
            f"verdict={v.verdict} inconsistent with test results "
            f"(tests_pass={_tests_pass(v)})"
        )
    return ValidationResult(ok=not errors, errors=errors)
