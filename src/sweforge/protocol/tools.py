"""Canonical 工具协议 — 五个统一工具及其结构化 action/observation。

    bash | search | view_file | str_replace | finish

Source Adapter 必须把源数据集工具映射到这五个；无法映射 drop，不伪造 observation。
SFT 与 RL 使用完全相同的这些结构（PROJECT_SPEC §2）。
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

__all__ = [
    "TOOL_NAMES",
    "ToolName",
    "BashAction",
    "SearchAction",
    "ViewFileAction",
    "StrReplaceAction",
    "FinishAction",
    "ToolAction",
    "BashObservation",
    "SearchMatch",
    "SearchObservation",
    "ViewFileObservation",
    "StrReplaceObservation",
    "FinishObservation",
    "ToolObservation",
    "parse_tool_action",
    "parse_tool_observation",
]

TOOL_NAMES = ("bash", "search", "view_file", "str_replace", "finish")
#: 注意: 不能用 Literal[TOOL_NAMES]（会把整个 tuple 当成单一字面量值）
ToolName = Literal["bash", "search", "view_file", "str_replace", "finish"]


# ============================ Actions（agent 发出） ============================


class BashAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["bash"] = "bash"
    command: str


class SearchAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["search"] = "search"
    query: str  # 仓库内文本/代码搜索（等价 grep -rn）
    path: str | None = None  # 可选：限定单个文件内搜索


class ViewFileAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["view_file"] = "view_file"
    path: str
    start_line: int | None = None  # 1-based，None = 文件开头
    end_line: int | None = None  # 含，None = 文件结尾

    @model_validator(mode="after")
    def _check_range(self) -> "ViewFileAction":
        if self.start_line is not None and self.end_line is not None:
            if self.start_line > self.end_line:
                raise ValueError(
                    f"start_line {self.start_line} > end_line {self.end_line}"
                )
        return self


class StrReplaceAction(BaseModel):
    """唯一匹配替换。empty old_string 定义插入/创建语义：

    old_string == "" -> 在文件开头插入 new_string；若文件不存在则以 new_string
    为内容创建该文件（RL 环境必须照此实现，PROJECT_SPEC §2）。
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["str_replace"] = "str_replace"
    path: str
    old_string: str  # 在文件中必须唯一匹配，否则 env 报错
    new_string: str


class FinishAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["finish"] = "finish"
    summary: str | None = None


ToolAction = Annotated[
    Union[BashAction, SearchAction, ViewFileAction, StrReplaceAction, FinishAction],
    Field(discriminator="name"),
]


# ========================= Observations（env 返回） ============================


class BashObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["bash"] = "bash"
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int | None = None


class SearchMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    line: int  # 1-based
    content: str


class SearchObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["search"] = "search"
    matches: list[SearchMatch] = Field(default_factory=list)
    truncated: bool = False  # 结果过多被截断
    error: str | None = None  # 搜索失败（无法执行等）


class ViewFileObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["view_file"] = "view_file"
    path: str
    start_line: int  # 1-based, 含（error 时为 0）
    end_line: int  # 含（error 时为 0）
    content: str  # 该窗口的渲染文本（按行）
    total_lines: int | None = None
    error: str | None = None  # 查看失败（路径不存在 / view_range 非法等）


class StrReplaceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["str_replace"] = "str_replace"
    success: bool
    error: str | None = None
    path: str | None = None


class FinishObservation(BaseModel):
    """finish 的确认观察（由 Agent Loop 生成，用于保持 turn 三元组一致）。"""

    model_config = ConfigDict(extra="forbid")

    name: Literal["finish"] = "finish"
    patch: str | None = None  # Agent Loop 导出的 unified diff 摘要


ToolObservation = Annotated[
    Union[
        BashObservation,
        SearchObservation,
        ViewFileObservation,
        StrReplaceObservation,
        FinishObservation,
    ],
    Field(discriminator="name"),
]

# Annotated 联合类型别名本身不是 pydantic 模型类，不能直接 .model_validate；
# 统一通过 TypeAdapter 解析（serialization / validate / 测试共用）。
_ACTION_ADAPTER = TypeAdapter(ToolAction)
_OBSERVATION_ADAPTER = TypeAdapter(ToolObservation)


def parse_tool_action(data: dict) -> ToolAction:
    return _ACTION_ADAPTER.validate_python(data)


def parse_tool_observation(data: dict) -> ToolObservation:
    return _OBSERVATION_ADAPTER.validate_python(data)
