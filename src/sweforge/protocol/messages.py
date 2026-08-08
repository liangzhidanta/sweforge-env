"""Canonical 消息 — tokenizer 输入的唯一消息格式。

消息顺序（PROJECT_SPEC §2）:
    system
    user/task                     （唯一 task 消息）
    assistant (+tool_calls)       （每条恰好 0 或 1 个 tool_call）
    tool (observation rendering)
    assistant (+tool_calls)
    tool
    ...
    assistant (+finish)           （终止；其后无 tool 消息）

tool 消息的 content 是结构化 observation 的确定性文本渲染
（protocol/serialization.py: render_observation）。
"""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sweforge.protocol.tools import ToolAction, ToolName

__all__ = ["CanonicalMessage", "CanonicalToolCall", "MessageRole", "ToolCallContent"]

MessageRole = Literal["system", "user", "assistant", "tool"]


class ToolCallContent(BaseModel):
    """tool_call 携带的结构化参数视图。

    canonical 内部存结构化 dict；与外部系统（OpenAI 风格 JSON 字符串）互转在
    serialization.py 完成。
    """

    model_config = ConfigDict(extra="forbid")

    action: ToolAction

    def arguments(self) -> dict[str, Any]:
        return self.action.model_dump(exclude={"name"})

    @classmethod
    def from_arguments(cls, name: str, arguments: dict[str, Any]) -> "ToolCallContent":
        from sweforge.protocol.serialization import action_from_name_args

        return cls(action=action_from_name_args(name, arguments))


class CanonicalToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str  # tool_call_id，消息内唯一
    name: ToolName
    arguments: dict[str, Any]  # 结构化参数（非 JSON 字符串）

    def as_action(self) -> ToolAction:
        from sweforge.protocol.serialization import action_from_name_args

        return action_from_name_args(self.name, self.arguments)


class CanonicalMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str | None = None
    #: role == "assistant" 时可带（恰 0 或 1 个）
    tool_calls: list[CanonicalToolCall] | None = None
    #: role == "tool" 时必须带，指向对应 assistant 消息的 tool_call.id
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _check_role_fields(self) -> "CanonicalMessage":
        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError("tool message requires tool_call_id")
            if self.tool_calls:
                raise ValueError("tool message must not carry tool_calls")
        elif self.role == "assistant":
            if self.tool_call_id is not None:
                raise ValueError("assistant message must not carry tool_call_id")
            if self.tool_calls is not None and len(self.tool_calls) > 1:
                raise ValueError(
                    "assistant message must carry at most one tool_call "
                    f"(got {len(self.tool_calls)})"
                )
        else:  # system / user
            if self.tool_calls is not None or self.tool_call_id is not None:
                raise ValueError(f"{self.role} message must not carry tool fields")
        return self

    @property
    def is_finish(self) -> bool:
        """该消息是否为 finish tool_call（轨迹终止标志）。"""
        return (
            self.role == "assistant"
            and self.tool_calls is not None
            and len(self.tool_calls) == 1
            and self.tool_calls[0].name == "finish"
        )
