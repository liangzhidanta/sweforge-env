"""SWE-Forge — 面向 Coding Agent 的工程级任务自动构造与可验证强化学习。

核心原则（见 PROJECT_SPEC.txt §0）:
    SFT 与 RL 共用同一种 Canonical AgentTrajectory 与 Tool Protocol。
"""

__version__ = "0.1.0"

# 顶层再导出，方便所有模块统一 import 路径。
from sweforge.protocol.tools import (  # noqa: F401
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
from sweforge.protocol.messages import (  # noqa: F401
    CanonicalMessage,
    CanonicalToolCall,
    MessageRole,
)
from sweforge.schemas.task import (  # noqa: F401
    MutationInfo,
    PolicyTask,
    TaskEnvironment,
    TaskSpec,
    TestSpec,
)
from sweforge.schemas.trajectory import (  # noqa: F401
    AgentTrajectory,
    AgentTurn,
    TerminationReason,
    RESPONSE_MASK_VALUE,
    PROMPT_MASK_VALUE,
)
from sweforge.schemas.verification import (  # noqa: F401
    TestResult,
    VerificationResult,
    binary_reward,
    is_resolved,
)
from sweforge.protocol.validate import (  # noqa: F401
    ValidationResult,
    validate_task_spec,
    validate_trajectory,
    validate_verification,
)
