"""VerificationResult — clean-container verifier 的产物（PROJECT_SPEC §5）。

resolved = 全部 F2P pass AND 全部 P2P pass AND integrity_ok
baseline reward = 1.0 / 0.0；shaped reward 仅用于 ablation（reward_detail 分项）。
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["TestResult", "VerificationResult", "binary_reward", "is_resolved"]


class TestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_id: str
    kind: Literal["fail_to_pass", "pass_to_pass"]
    passed: bool
    duration_ms: int | None = None
    output: str | None = None  # 截断后的尾部输出，仅调试用


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verification_id: str
    task_id: str
    patch: str | None = None
    verdict: Literal["resolved", "unresolved", "error"]

    fail_to_pass: list[TestResult] = Field(default_factory=list)
    pass_to_pass: list[TestResult] = Field(default_factory=list)

    integrity_ok: bool = True

    #: baseline: resolved -> 1.0；shaped 由训练 config 决定
    reward: float | None = None
    reward_detail: dict[str, Any] | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.verdict == "resolved"

    @property
    def f2p_passed(self) -> int:
        return sum(1 for r in self.fail_to_pass if r.passed)

    @property
    def p2p_passed(self) -> int:
        return sum(1 for r in self.pass_to_pass if r.passed)


def _tests_pass(v: VerificationResult) -> bool:
    """纯测试逻辑（不含 verdict）：全部 F2P pass AND 全部 P2P pass AND integrity_ok。"""
    return (
        bool(v.fail_to_pass)
        and all(r.passed for r in v.fail_to_pass)
        and all(r.passed for r in v.pass_to_pass)
        and v.integrity_ok
    )


def is_resolved(v: VerificationResult) -> bool:
    """resolved = verdict 为 resolved 且测试/完整性全过（reward 语义）。

    verdict 与 _tests_pass 的不一致是协议 bug，由 validate_verification() 双向校验。
    """
    return v.verdict == "resolved" and _tests_pass(v)


def binary_reward(v: VerificationResult) -> float:
    """主 baseline reward：1.0 / 0.0（以 verdict 为准）。"""
    return 1.0 if is_resolved(v) else 0.0
