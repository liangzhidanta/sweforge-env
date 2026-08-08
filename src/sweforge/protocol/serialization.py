"""Canonical 序列化 — 消息渲染 / token mask 组装 / OpenAI 风格互转 / JSONL 存取。

职责:
    1. render_observation: 结构化 observation -> 确定性文本（tool 消息 content）
    2. build_token_arrays:  按 mask 语义（system/user/tool=0, assistant=1）
                            从每条消息的 token 数组装 prompt_ids / response_ids /
                            response_mask（tokenizer 无关）
    3. messages_to_openai / canonical_messages_from_openai:
                            vLLM / veRL 的 OpenAI 风格消息互转（RL rollout 回灌）
    4. trajectory JSONL 存取（SFT 数据集 / rollout 存储共用）

RL 原则（PROJECT_SPEC §0 P2）: rollout 保留真实 token 序列，禁止事后重新
tokenize 替代。消息边界通过 AgentTrajectory.message_token_counts 保留。
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from sweforge.protocol.messages import CanonicalMessage, CanonicalToolCall
from sweforge.protocol.tools import (
    TOOL_NAMES,
    BashObservation,
    FinishObservation,
    SearchObservation,
    StrReplaceObservation,
    ToolAction,
    ToolObservation,
    ViewFileObservation,
    parse_tool_action,
)

__all__ = [
    "MAX_OBSERVATION_CHARS",
    "TokenArrays",
    "action_from_name_args",
    "build_token_arrays",
    "canonical_messages_from_openai",
    "messages_to_openai",
    "render_message",
    "render_observation",
    "render_turns_to_messages",
    "trajectories_from_jsonl",
    "trajectory_to_jsonl",
]

#: 单条 observation 渲染进消息的最大字符数（超出截断，保证 context 可控）
MAX_OBSERVATION_CHARS = 4000

_ACTION_NAMES = {
    "bash": "BashAction",
    "search": "SearchAction",
    "view_file": "ViewFileAction",
    "str_replace": "StrReplaceAction",
    "finish": "FinishAction",
}

_OBSERVATION_NAMES = {
    "bash": "BashObservation",
    "search": "SearchObservation",
    "view_file": "ViewFileObservation",
    "str_replace": "StrReplaceObservation",
    "finish": "FinishObservation",
}


def action_from_name_args(name: str, arguments: dict[str, Any]) -> ToolAction:
    """按工具名 + 结构化参数构造 ToolAction（未知工具抛 ValueError -> drop）。"""
    if name not in _ACTION_NAMES:
        raise ValueError(f"unknown tool name: {name!r} (allowed: {TOOL_NAMES})")
    return parse_tool_action({**arguments, "name": name})


# ============================ 1. observation 渲染 ============================


def _truncate(text: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def render_observation(
    obs: ToolObservation | None, limit: int = MAX_OBSERVATION_CHARS
) -> str:
    """结构化 observation -> 确定性文本（tool 消息的 content）。

    同一 observation 在任何流程（SFT 归一化 / RL rollout / 验证校验）中渲染结果
    必须一致，validate_trajectory() 依赖此性质。
    """
    if obs is None:
        return ""
    if isinstance(obs, BashObservation):
        parts = []
        if obs.stdout:
            parts.append(obs.stdout.rstrip("\n"))
        if obs.stderr:
            parts.append(f"[stderr]\n{obs.stderr.rstrip()}")
        tail = _truncate("\n".join(parts), limit)
        return f"[exit code {obs.exit_code}]\n{tail}" if tail else f"[exit code {obs.exit_code}]"
    if isinstance(obs, SearchObservation):
        if obs.error:
            return f"[search failed] {obs.error}"
        if not obs.matches:
            return "[no matches]"
        lines = [f"{m.path}:{m.line}: {m.content}" for m in obs.matches]
        if obs.truncated:
            lines.append("...[search truncated]")
        return _truncate("\n".join(lines), limit)
    if isinstance(obs, ViewFileObservation):
        if obs.error:
            return f"[view_file failed] {obs.error}"
        lines = [
            f"{obs.path}:{obs.start_line}-{obs.end_line}"
            + (f" (total {obs.total_lines} lines)" if obs.total_lines is not None else "")
        ]
        lines.extend(obs.content.splitlines())
        return _truncate("\n".join(lines), limit)
    if isinstance(obs, StrReplaceObservation):
        if obs.success:
            return f"[str_replace applied] {obs.path}"
        return f"[str_replace failed] {obs.error or 'unknown error'}"
    if isinstance(obs, FinishObservation):
        return "[finish]"
    raise TypeError(f"unknown observation type: {type(obs)}")


def render_message(msg: CanonicalMessage, limit: int = MAX_OBSERVATION_CHARS) -> dict[str, Any]:
    """CanonicalMessage -> OpenAI 风格 dict（vLLM / veRL chat 输入）。

    content=None 归一化为 ""（Qwen3 chat template 对 content=None 的 assistant
    消息执行 '</think>' in None 会抛 TypeError；纯工具调用消息没有文字注释）。
    """
    out: dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
    if msg.role == "assistant" and msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in msg.tool_calls
        ]
    if msg.role == "tool":
        out["tool_call_id"] = msg.tool_call_id
    return out


def messages_to_openai(
    messages: Iterable[CanonicalMessage], limit: int = MAX_OBSERVATION_CHARS
) -> list[dict[str, Any]]:
    return [render_message(m, limit=limit) for m in messages]


def canonical_messages_from_openai(records: Iterable[dict[str, Any]]) -> list[CanonicalMessage]:
    """OpenAI 风格消息（vLLM / veRL 输出）-> CanonicalMessage。

    无法解析（未知工具名 / 非法 JSON 参数）抛 ValueError，调用方 drop 整条轨迹。
    """
    out: list[CanonicalMessage] = []
    for rec in records:
        role = rec["role"]
        if role == "assistant" and rec.get("tool_calls"):
            calls = []
            for tc in rec["tool_calls"]:
                fn = tc["function"]
                args = json.loads(fn["arguments"])
                calls.append(
                    CanonicalToolCall(id=tc["id"], name=fn["name"], arguments=args)
                )
            out.append(CanonicalMessage(role="assistant", content=rec.get("content"), tool_calls=calls))
        elif role == "assistant":
            out.append(CanonicalMessage(role="assistant", content=rec.get("content")))
        elif role == "tool":
            out.append(
                CanonicalMessage(
                    role="tool", content=rec.get("content"), tool_call_id=rec["tool_call_id"]
                )
            )
        elif role in ("system", "user"):
            out.append(CanonicalMessage(role=role, content=rec.get("content")))
        else:
            raise ValueError(f"unsupported role in OpenAI records: {role!r}")
    return out


def render_turns_to_messages(turns: Iterable[Any]) -> list[CanonicalMessage]:
    """从 turns（AgentTurn 列表）重建 messages —— 校验用参考实现。

    validate_trajectory() 用它检查存储的 messages 与结构化 turns 一致。
    """
    messages: list[CanonicalMessage] = []
    for turn in turns:
        messages.append(turn.assistant_message)
        if turn.observation is not None:
            messages.append(
                CanonicalMessage(
                    role="tool",
                    content=render_observation(turn.observation),
                    tool_call_id=turn.assistant_message.tool_calls[0].id,
                )
            )
    return messages


# ============================ 2. token mask 组装 ============================


class TokenArrays:
    """build_token_arrays 的产物：统一的 token / loss mask 三元组。"""

    __slots__ = ("prompt_ids", "response_ids", "response_mask")

    def __init__(self, prompt_ids: list[int], response_ids: list[int], response_mask: list[int]):
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.response_mask = response_mask


def build_token_arrays(
    token_segments: list[list[int]],
    roles: list[str],
    response_mask_values: list[int] | None = None,
    assistant_prefix_mask: int = 0,
) -> TokenArrays:
    """按统一 mask 语义组装 token 数组（PROJECT_SPEC §0 P2 / §3）。

    Args:
        token_segments: 与消息一一对应的真实 token 段（SFT 用 tokenizer 分词结果，
                        RL 用 rollout 保留的真实 token 序列）
        roles:          与消息一一对应的 role
        response_mask_values: 可选，与拼接后的 response_ids 等长；默认全 1
                             （所有 assistant 文本都是学习目标，包括 tool_call
                             语法本身）
        assistant_prefix_mask: 每个 assistant 段的开头 N 个 token mask=0
                             （veRL SFT 语义: chat 模板注入的 generation prompt
                             "<|im_start|>assistant\\n" 不是模型该学习的文本；
                             RL 路径默认 0 = 全监督，不受影响）
    """
    if len(token_segments) != len(roles):
        raise ValueError("token_segments and roles must be aligned")
    if assistant_prefix_mask < 0:
        raise ValueError(f"assistant_prefix_mask must be >= 0: {assistant_prefix_mask}")

    prompt_ids: list[int] = []
    response_ids: list[int] = []
    response_mask: list[int] = []

    for segment, role in zip(token_segments, roles):
        if role == "assistant":
            response_ids.extend(segment)
            prefix = min(assistant_prefix_mask, len(segment))
            response_mask.extend([0] * prefix)
            response_mask.extend([1] * (len(segment) - prefix))
        else:
            prompt_ids.extend(segment)

    if response_mask_values is not None:
        if len(response_mask_values) != len(response_ids):
            raise ValueError(
                f"response_mask_values length {len(response_mask_values)} != "
                f"response_ids length {len(response_ids)}"
            )
        if any(v not in (0, 1) for v in response_mask_values):
            raise ValueError(f"response_mask_values must be 0/1: {response_mask_values}")
        response_mask = list(response_mask_values)
    return TokenArrays(prompt_ids=prompt_ids, response_ids=response_ids, response_mask=response_mask)


# ============================ 3. JSONL 存取 ============================


def trajectory_to_jsonl(trajectory: Any, path: str) -> None:
    """AgentTrajectory -> JSONL（model_dump(mode="json")，可安全 round-trip）。"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trajectory.model_dump(mode="json"), ensure_ascii=False) + "\n")


def trajectories_from_jsonl(path: str) -> list[Any]:
    from sweforge.schemas.trajectory import AgentTrajectory

    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(AgentTrajectory.model_validate_json(line))
    return out
