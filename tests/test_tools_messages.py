"""Tool Protocol 单元测试：五工具 discriminated union + 消息 role 规则。"""

import pytest
from pydantic import ValidationError

from sweforge.protocol.messages import CanonicalMessage, CanonicalToolCall
from sweforge.protocol.tools import (
    TOOL_NAMES,
    BashAction,
    FinishAction,
    SearchAction,
    StrReplaceAction,
    ToolObservation,
    ViewFileAction,
    parse_tool_action,
    parse_tool_observation,
)


class TestToolActions:
    def test_tool_names_fixed(self):
        assert TOOL_NAMES == ("bash", "search", "view_file", "str_replace", "finish")

    def test_discriminated_union_roundtrip(self):
        for action in [
            BashAction(command="pwd"),
            SearchAction(query="def foo"),
            ViewFileAction(path="a.py", start_line=1, end_line=10),
            StrReplaceAction(path="a.py", old_string="x", new_string="y"),
            FinishAction(summary="done"),
        ]:
            d = action.model_dump()
            assert "name" in d
            rebuilt = parse_tool_action(d)
            assert rebuilt == action
            assert type(rebuilt) is type(action)

    def test_unknown_tool_name_rejected(self):
        with pytest.raises(ValidationError):
            parse_tool_action({"name": "rm_rf", "command": "echo hi"})

    def test_view_file_range(self):
        with pytest.raises(ValidationError):
            ViewFileAction(path="a.py", start_line=10, end_line=5)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            BashAction(command="pwd", sneaky=True)


class TestObservations:
    def test_observation_union_roundtrip(self):
        obs = parse_tool_observation(
            {"name": "bash", "exit_code": 0, "stdout": "ok", "stderr": ""}
        )
        # ToolObservation 是 Annotated 别名，不能用于 isinstance；按判别字段断言
        assert obs.name == "bash"
        assert type(obs).__name__ == "BashObservation"

    def test_observation_discriminated(self):
        with pytest.raises(ValidationError):
            parse_tool_observation({"name": "bash"})  # 缺 exit_code


class TestCanonicalMessage:
    def test_tool_message_requires_call_id(self):
        with pytest.raises(ValidationError):
            CanonicalMessage(role="tool", content="out")

    def test_assistant_at_most_one_call(self):
        with pytest.raises(ValidationError):
            CanonicalMessage(
                role="assistant",
                content="x",
                tool_calls=[
                    CanonicalToolCall(id="a", name="bash", arguments={"command": "1"}),
                    CanonicalToolCall(id="b", name="bash", arguments={"command": "2"}),
                ],
            )

    def test_user_message_no_tool_fields(self):
        with pytest.raises(ValidationError):
            CanonicalMessage(role="user", content="hi", tool_call_id="x")

    def test_is_finish(self):
        m = CanonicalMessage(
            role="assistant",
            content="done",
            tool_calls=[CanonicalToolCall(id="f", name="finish", arguments={})],
        )
        assert m.is_finish
        m2 = CanonicalMessage(
            role="assistant",
            content="run",
            tool_calls=[CanonicalToolCall(id="b", name="bash", arguments={"command": "pwd"})],
        )
        assert not m2.is_finish
