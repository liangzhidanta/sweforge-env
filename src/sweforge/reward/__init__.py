"""reward/ —— verifier 客户端 + reward 计算（PROJECT_SPEC §1）。

阶段 9: CleanVerifier（reward/verifier.py）; reward 语义见
schemas/verification.py（binary_reward = 主 baseline reward）。
"""

from sweforge.reward.verifier import (
    CleanVerifier,
    DockerIsolation,
    Isolation,
    TempDirIsolation,
    VERIFIER_VERSION,
)

__all__ = [
    "CleanVerifier",
    "DockerIsolation",
    "Isolation",
    "TempDirIsolation",
    "VERIFIER_VERSION",
]
