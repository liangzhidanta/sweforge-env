"""TaskSpec — RL 任务池的存储单元（不是 expert trajectory）。

Policy-visible 视图（policy_view()）严格剥离: gold patch / hidden test /
mutation location / original repair / verifier secret（PROJECT_SPEC §0 P5）。

Data Factory 产出 Verified Task 后，TaskSpec 进入任务池；训练与 rollout
只使用 policy_view() 结果。
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Difficulty",
    "MutationInfo",
    "PolicyTask",
    "TaskEnvironment",
    "TaskSpec",
    "TestSpec",
]

Difficulty = Literal["easy", "mixed", "currently_unsolved"]


class TestSpec(BaseModel):
    """单个测试的规格。test_id 是 environment.test_commands 的键。"""

    model_config = ConfigDict(extra="forbid")

    test_id: str
    kind: Literal["fail_to_pass", "pass_to_pass"]
    timeout_seconds: int = 300
    notes: str | None = None  # 仅内部记账用，policy 不可见


class TaskEnvironment(BaseModel):
    """构建/测试环境规格。image 由 Mac 侧解析为实际 docker image。"""

    model_config = ConfigDict(extra="forbid")

    image: str | None = None
    setup_commands: list[str] = Field(default_factory=list)
    build_commands: list[str] = Field(default_factory=list)
    #: test_id -> 运行该测试的 shell 命令
    test_commands: dict[str, list[str]] = Field(default_factory=dict)


class MutationInfo(BaseModel):
    """Data Factory 记账信息。仅 server 侧，禁止进入 policy_view。"""

    model_config = ConfigDict(extra="forbid")

    kind: str  # "real_repair_reversal" | "invert_condition" | "operator_mutation" | ...
    source_commit: str | None = None
    source_pr: str | None = None
    file: str | None = None
    location: str | None = None
    recipe_seed: int | None = None


class TaskSpec(BaseModel):
    """完整任务规格（server 侧）。训练/rollout 使用 policy_view()。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    repo: str
    base_commit: str
    problem_statement: str
    environment: TaskEnvironment = Field(default_factory=TaskEnvironment)

    fail_to_pass: list[TestSpec] = Field(default_factory=list)
    pass_to_pass: list[TestSpec] = Field(default_factory=list)

    gold_patch: str | None = None
    mutation: MutationInfo | None = None

    difficulty: Difficulty | None = None  # 由 Difficulty Profiler 设置
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_ids_unique(self) -> "TaskSpec":
        seen: set[str] = set()
        for t in [*self.fail_to_pass, *self.pass_to_pass]:
            if t.test_id in seen:
                raise ValueError(f"duplicate test_id: {t.test_id!r}")
            seen.add(t.test_id)
        return self

    def policy_view(self) -> "PolicyTask":
        """构造 policy 可见视图：结构性 redact（不抛错）。

        PolicyTask 只有 5 个字段，gold_patch / fail_to_pass / pass_to_pass /
        mutation 结构上不可能进入视图；任务池卫生检查（secret 必须已剥离）由
        protocol.validate.validate_task_spec() 负责。
        environment 深拷贝后剥离 test_commands（连测试数量/名称都不得泄漏）。
        """
        env = self.environment.model_copy(deep=True)
        env.test_commands = {}
        return PolicyTask(
            task_id=self.task_id,
            repo=self.repo,
            base_commit=self.base_commit,
            problem_statement=self.problem_statement,
            environment=env,
        )


class PolicyTask(BaseModel):
    """Policy 可见的任务视图（TaskSpec.policy_view() 的产物）。

    environment 中剥离 test_commands：连测试数量/名称都不得泄漏。
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    repo: str
    base_commit: str
    problem_statement: str
    environment: TaskEnvironment

    @model_validator(mode="after")
    def _no_test_commands(self) -> "PolicyTask":
        if self.environment.test_commands:
            raise ValueError("PolicyTask.environment must not carry test_commands")
        return self
