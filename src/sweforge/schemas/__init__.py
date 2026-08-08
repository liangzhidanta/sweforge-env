"""Canonical schemas: TaskSpec / AgentTrajectory / VerificationResult。

SFT 与 RL 共用这一套结构，禁止为两侧分别设计格式（PROJECT_SPEC §0 P1）。
"""

from sweforge.schemas.task import MutationInfo, PolicyTask, TaskEnvironment, TaskSpec, TestSpec
from sweforge.schemas.trajectory import (
    PROMPT_MASK_VALUE,
    RESPONSE_MASK_VALUE,
    AgentTrajectory,
    AgentTurn,
    TerminationReason,
)
from sweforge.schemas.verification import (
    TestResult,
    VerificationResult,
    binary_reward,
    is_resolved,
)

__all__ = [
    "AgentTrajectory",
    "AgentTurn",
    "MutationInfo",
    "PolicyTask",
    "PROMPT_MASK_VALUE",
    "RESPONSE_MASK_VALUE",
    "TaskEnvironment",
    "TaskSpec",
    "TerminationReason",
    "TestResult",
    "TestSpec",
    "VerificationResult",
    "binary_reward",
    "is_resolved",
]
