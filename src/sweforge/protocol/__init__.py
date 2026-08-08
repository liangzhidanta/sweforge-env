"""Canonical Tool Protocol — SFT 与 RL 共用的消息/工具/序列化/校验。

任何公开数据进入系统前必须经过 Source Adapter 归一化；无法映射的工具 drop，
不伪造 observation（PROJECT_SPEC §2）。
"""

from sweforge.protocol.messages import CanonicalMessage, CanonicalToolCall, MessageRole
from sweforge.protocol.system_prompt import (
    CANONICAL_SYSTEM_PROMPT,
    CANONICAL_SYSTEM_PROMPT_VERSION,
)
from sweforge.protocol.serialization import (
    MAX_OBSERVATION_CHARS,
    build_token_arrays,
    canonical_messages_from_openai,
    messages_to_openai,
    render_message,
    render_observation,
    render_turns_to_messages,
    trajectories_from_jsonl,
    trajectory_to_jsonl,
)
from sweforge.protocol.tools import (
    TOOL_NAMES,
    BashAction,
    BashObservation,
    FinishAction,
    FinishObservation,
    SearchAction,
    SearchMatch,
    SearchObservation,
    StrReplaceAction,
    StrReplaceObservation,
    ToolAction,
    ToolObservation,
    ViewFileAction,
    ViewFileObservation,
)
from sweforge.protocol.validate import (
    ValidationResult,
    validate_task_spec,
    validate_trajectory,
    validate_verification,
)

__all__ = [
    "BashAction",
    "BashObservation",
    "CANONICAL_SYSTEM_PROMPT",
    "CANONICAL_SYSTEM_PROMPT_VERSION",
    "CanonicalMessage",
    "CanonicalToolCall",
    "FinishAction",
    "FinishObservation",
    "MAX_OBSERVATION_CHARS",
    "MessageRole",
    "SearchAction",
    "SearchMatch",
    "SearchObservation",
    "StrReplaceAction",
    "StrReplaceObservation",
    "TOOL_NAMES",
    "ToolAction",
    "ToolObservation",
    "ValidationResult",
    "ViewFileAction",
    "ViewFileObservation",
    "build_token_arrays",
    "canonical_messages_from_openai",
    "messages_to_openai",
    "render_message",
    "render_observation",
    "render_turns_to_messages",
    "trajectories_from_jsonl",
    "trajectory_to_jsonl",
    "validate_task_spec",
    "validate_trajectory",
    "validate_verification",
]
