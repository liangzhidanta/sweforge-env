"""序列化与 token/mask 语义测试。"""

import json

import pytest

from sweforge.protocol.messages import CanonicalMessage, CanonicalToolCall
from sweforge.protocol.serialization import (
    MAX_OBSERVATION_CHARS,
    build_token_arrays,
    canonical_messages_from_openai,
    messages_to_openai,
    render_message,
    render_observation,
    render_turns_to_messages,
    trajectory_to_jsonl,
    trajectories_from_jsonl,
)
from sweforge.protocol.tools import (
    BashObservation,
    SearchMatch,
    SearchObservation,
    StrReplaceObservation,
    ViewFileObservation,
)
from sweforge.schemas.trajectory import TerminationReason
from tests.helpers import build_valid_trajectory


class TestRenderObservation:
    def test_bash(self):
        obs = BashObservation(exit_code=1, stdout="FAIL x", stderr="trace")
        assert render_observation(obs) == "[exit code 1]\nFAIL x\n[stderr]\ntrace"

    def test_bash_empty(self):
        obs = BashObservation(exit_code=0, stdout="", stderr="")
        assert render_observation(obs) == "[exit code 0]"

    def test_search(self):
        obs = SearchObservation(
            matches=[SearchMatch(path="a.py", line=3, content="def foo():")],
            truncated=True,
        )
        assert render_observation(obs) == "a.py:3: def foo():\n...[search truncated]"

    def test_view_file(self):
        obs = ViewFileObservation(
            path="a.py", start_line=1, end_line=2, content="line1\nline2", total_lines=100
        )
        assert render_observation(obs).startswith("a.py:1-2 (total 100 lines)\nline1\nline2")

    def test_str_replace(self):
        assert render_observation(StrReplaceObservation(success=True, path="a.py")) == (
            "[str_replace applied] a.py"
        )
        assert render_observation(StrReplaceObservation(success=False, error="not unique")) == (
            "[str_replace failed] not unique"
        )

    def test_truncation(self):
        obs = BashObservation(exit_code=0, stdout="x" * (MAX_OBSERVATION_CHARS + 1000))
        rendered = render_observation(obs)
        assert len(rendered) <= MAX_OBSERVATION_CHARS + 100
        assert "[truncated" in rendered


class TestTokenArrays:
    def test_mask_semantics(self):
        # system(2) user(4) | assistant(3) tool(2) assistant(3) finish-assistant(1)
        segments = [[0, 0], [0, 0, 0, 0], [10, 11, 12], [20, 21], [30, 31, 32], [40]]
        roles = ["system", "user", "assistant", "tool", "assistant", "assistant"]
        arr = build_token_arrays(segments, roles)
        assert arr.prompt_ids == [0, 0, 0, 0, 0, 0, 20, 21]
        assert arr.response_ids == [10, 11, 12, 30, 31, 32, 40]
        assert arr.response_mask == [1] * 7

    def test_custom_mask(self):
        segments = [[1, 2, 3]]
        roles = ["assistant"]
        arr = build_token_arrays(segments, roles, response_mask_values=[1, 0, 1])
        assert arr.response_mask == [1, 0, 1]

    def test_assistant_prefix_mask(self):
        """veRL 对齐: 每条 assistant 段开头 N token mask=0（generation prompt 不学习）。"""
        segments = [[1, 2, 3], [10, 11]]
        roles = ["assistant", "assistant"]
        arr = build_token_arrays(segments, roles, assistant_prefix_mask=2)
        assert arr.response_ids == [1, 2, 3, 10, 11]
        assert arr.response_mask == [0, 0, 1, 0, 0]
        # 默认 0 = 全监督（RL 路径语义不变）
        assert build_token_arrays(segments, roles).response_mask == [1] * 5

    def test_prefix_mask_clamps_short_segment(self):
        arr = build_token_arrays([[5]], ["assistant"], assistant_prefix_mask=4)
        assert arr.response_mask == [0]

    def test_prefix_mask_rejects_negative(self):
        with pytest.raises(ValueError):
            build_token_arrays([[1]], ["assistant"], assistant_prefix_mask=-1)

    def test_mask_length_mismatch(self):
        with pytest.raises(ValueError):
            build_token_arrays([[1, 2]], ["assistant"], response_mask_values=[1])

    def test_alignment(self):
        with pytest.raises(ValueError):
            build_token_arrays([[1]], ["system", "user"])


class TestOpenaiMapping:
    def test_roundtrip(self):
        m1 = CanonicalMessage(role="system", content="sys")
        m2 = CanonicalMessage(role="user", content="task")
        m3 = CanonicalMessage(
            role="assistant",
            content="run",
            tool_calls=[
                CanonicalToolCall(
                    id="c1", name="bash", arguments={"command": "pytest -q"}
                )
            ],
        )
        m4 = CanonicalMessage(role="tool", content="[exit code 0]", tool_call_id="c1")
        out = messages_to_openai([m1, m2, m3, m4])
        assert out[2]["tool_calls"][0]["function"]["arguments"] == '{"command": "pytest -q"}'
        back = canonical_messages_from_openai(out)
        assert back == [m1, m2, m3, m4]

    def test_rejects_unknown_role(self):
        with pytest.raises(ValueError):
            canonical_messages_from_openai([{"role": "function", "content": "x"}])

    def test_render_message_tool(self):
        d = render_message(CanonicalMessage(role="tool", content="out", tool_call_id="c9"))
        assert d == {"role": "tool", "content": "out", "tool_call_id": "c9"}

    def test_render_message_none_content_becomes_empty(self):
        """content=None 必须渲染为 ""（Qwen3 模板对 None 的 '</think>' in 检查会崩）。"""
        m = CanonicalMessage(
            role="assistant",
            content=None,
            tool_calls=[CanonicalToolCall(id="c1", name="bash", arguments={"command": "ls"})],
        )
        d = render_message(m)
        assert d["content"] == ""
        assert d["tool_calls"][0]["function"]["name"] == "bash"


class TestTurnsRender:
    def test_render_matches_fixture(self):
        from tests.helpers import build_messages_and_turns

        messages, turns = build_messages_and_turns()
        # turns 只渲染 assistant/tool 部分；与 messages 中对应部分一致
        core = [m for m in messages if m.role in ("assistant", "tool")]
        assert render_turns_to_messages(turns) == core


class TestJsonlRoundtrip:
    def test_roundtrip(self, tmp_path):
        t = build_valid_trajectory()
        p = tmp_path / "traj.jsonl"
        trajectory_to_jsonl(t, p)
        trajectory_to_jsonl(build_valid_trajectory(trajectory_id="traj_0002"), p)
        back = trajectories_from_jsonl(p)
        assert len(back) == 2
        assert back[0] == t
        assert back[0].termination_reason == TerminationReason.AGENT_FINISHED
        # json 字段能完整 round-trip
        raw = json.loads(p.read_text().splitlines()[0])
        assert raw["metadata"]["protocol_version"] == "canonical-v1"
