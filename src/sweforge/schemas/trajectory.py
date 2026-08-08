"""Canonical AgentTrajectory — SFT 与 RL 共用的唯一轨迹格式（PROJECT_SPEC §3）。

结构:
    messages                tokenizer 输入的真实消息序列（system/user/assistant/tool）
    turns                   结构化语义视图（assistant_message + action + observation）
    prompt_ids / response_ids / response_mask
                            统一的 token / loss mask 语义:
                            system/user/tool = mask 0（context）
                            assistant        = mask 1（SFT loss / RL policy loss）
    message_token_counts    与 messages 等长；不重新 tokenize 即可恢复消息边界
    logprobs / reward / verification
                            仅 RL 有值

一致性由 protocol/validate.py 的 validate_trajectory() 强制（§10 验收硬门槛）。
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sweforge.protocol.messages import CanonicalMessage
from sweforge.protocol.tools import ToolAction, ToolObservation

__all__ = [
    "PROMPT_MASK_VALUE",
    "RESPONSE_MASK_VALUE",
    "AgentTrajectory",
    "AgentTurn",
    "TerminationReason",
]

#: token / loss mask 常量：环境 observation 永远是 context（PROJECT_SPEC §0 P2）
PROMPT_MASK_VALUE = 0
RESPONSE_MASK_VALUE = 1


class TerminationReason(str, Enum):
    AGENT_FINISHED = "agent_finished"  # 最后一条消息是 finish tool_call
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    TIME_LIMIT_EXCEEDED = "time_limit_exceeded"
    AGENT_ERROR = "agent_error"  # 模型输出无法解析（非法 JSON / 未知工具）
    ENVIRONMENT_ERROR = "environment_error"
    INTERRUPTED = "interrupted"


class AgentTurn(BaseModel):
    """一个 agent 回合：assistant 消息 + 结构化 action + observation。"""

    model_config = ConfigDict(extra="forbid")

    assistant_message: CanonicalMessage
    action: ToolAction
    #: finish 或异常终止时为 None（此时 messages 中该 assistant 消息后没有 tool 消息）
    observation: ToolObservation | None = None


class AgentTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory_id: str
    task_id: str

    # ---- 语义层 ----
    messages: list[CanonicalMessage]
    turns: list[AgentTurn]

    # ---- token 层（真实 token 序列，禁止事后重新 tokenize 替代）----
    prompt_ids: list[int] = Field(default_factory=list)
    response_ids: list[int] = Field(default_factory=list)
    response_mask: list[int] = Field(default_factory=list)
    #: 与 messages 等长：每条消息的 token 数；sum == len(prompt_ids) + len(response_ids)
    message_token_counts: list[int] | None = None

    termination_reason: TerminationReason
    patch: str | None = None  # unified diff（final vs base_commit）

    # ---- 仅 RL ----
    reward: float | None = None
    verification: Any | None = None  # VerificationResult（schemas.verification）
    logprobs: list[float] | None = None  # 与 response_ids 等长

    # ---- 元数据 ----
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _verify_token_shapes(self) -> "AgentTrajectory":
        if len(self.response_mask) != len(self.response_ids):
            raise ValueError(
                f"response_mask length {len(self.response_mask)} != "
                f"response_ids length {len(self.response_ids)}"
            )
        if self.logprobs is not None and len(self.logprobs) != len(self.response_ids):
            raise ValueError(
                f"logprobs length {len(self.logprobs)} != response_ids length {len(self.response_ids)}"
            )
        if self.message_token_counts is not None:
            if len(self.message_token_counts) != len(self.messages):
                raise ValueError(
                    f"message_token_counts length {len(self.message_token_counts)} "
                    f"!= messages length {len(self.messages)}"
                )
            if sum(self.message_token_counts) != len(self.prompt_ids) + len(self.response_ids):
                raise ValueError(
                    "sum(message_token_counts) != len(prompt_ids) + len(response_ids): "
                    f"{sum(self.message_token_counts)} != {len(self.prompt_ids) + len(self.response_ids)}"
                )
        return self

    # ---- 常用统计 ----

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_ids)

    @property
    def num_response_tokens(self) -> int:
        return len(self.response_ids)

    @property
    def loss_token_count(self) -> int:
        """mask=1 的 token 数（SFT loss / RL policy loss 的覆盖量）。"""
        return sum(self.response_mask)

    def labels_for_sft(self, ignore_index: int = -100) -> list[int]:
        """SFT 因果 loss 的 label 序列（与完整输入序列等长）。

        完整序列 = 按 message_token_counts 边界交替拼接 prompt_ids 与 response_ids；
        此处直接给出等价 label 序列：prompt 区 -> ignore_index，response 区 -> token id。
        """
        if self.message_token_counts is None:
            raise ValueError("labels_for_sft requires message_token_counts")
        labels: list[int] = []
        p = r = 0
        for msg_tokens, msg in zip(self.message_token_counts, self.messages):
            if msg.role == "assistant":
                labels.extend(self.response_ids[r : r + msg_tokens])
                r += msg_tokens
            else:
                labels.extend([ignore_index] * msg_tokens)
                p += msg_tokens
        return labels
