from sweforge.env_server.docker.executors import LocalExecutor
from sweforge.env_server.docker.tools import execute_action
from sweforge.schemas import ToolAction


def _executor(tmp_path):
    (tmp_path / "a.py").write_text("def get_or_compute(key):\n    return key\n")
    return LocalExecutor(tmp_path)


def test_bash_success(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("bash", {"command": "echo ok"}, request_id="r1"), "e1")
    assert observation.tool == "bash" and observation.exit_code == 0
    assert observation.stdout.strip() == "ok"
    assert observation.request_id == "r1" and observation.env_id == "e1"


def test_bash_empty_command(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("bash", {"command": "  "}, request_id="r1"), "e1")
    assert observation.exit_code is None and "command cannot be empty" in observation.content


def test_bash_timeout(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("bash", {"command": "sleep 5", "timeout": 0.2}, request_id="r1"), "e1")
    assert observation.exit_code == 124


def test_unknown_tool(tmp_path):
    observation = execute_action(_executor(tmp_path), ToolAction("curl", {}, request_id="r1"), "e1")
    assert observation.exit_code is None and "unknown tool" in observation.content


def test_search_found(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("search", {"pattern": "get_or_compute"}, request_id="r1"), "e1")
    assert observation.exit_code == 0 and "a.py:1" in observation.content


def test_search_no_match(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("search", {"pattern": "zzz_no_match"}, request_id="r1"), "e1")
    assert observation.exit_code == 1 and observation.content == "NO_MATCHES"


def test_search_max_results(tmp_path):
    executor = LocalExecutor(tmp_path)
    executor.write_text("b.txt", "\n".join(f"needle {i}" for i in range(100)))
    observation = execute_action(executor,
                                 ToolAction("search", {"pattern": "needle", "max_results": 5}, request_id="r1"), "e1")
    assert len(observation.content.splitlines()) == 5


def test_view_file_range(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("view_file", {"path": "a.py", "start_line": 1, "end_line": 2}, request_id="r1"), "e1")
    assert observation.exit_code == 0 and "| def get_or_compute" in observation.content


def test_view_file_invalid_range(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("view_file", {"path": "a.py", "start_line": 5, "end_line": 2}, request_id="r1"), "e1")
    assert observation.exit_code is None and "invalid line range" in observation.content


def test_view_file_escape(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("view_file", {"path": "../secret.txt"}, request_id="r1"), "e1")
    assert observation.exit_code is None and "escapes" in observation.content


def test_str_replace_single(tmp_path):
    executor = _executor(tmp_path)
    observation = execute_action(executor,
                                 ToolAction("str_replace", {"path": "a.py", "old": "return key",
                                                            "new": "return key.upper()", "expected_occurrences": 1}, request_id="r1"), "e1")
    assert observation.exit_code == 0
    assert executor.read_text("a.py") == "def get_or_compute(key):\n    return key.upper()\n"


def test_str_replace_zero(tmp_path):
    observation = execute_action(_executor(tmp_path),
                                 ToolAction("str_replace", {"path": "a.py", "old": "absent", "new": "x"}, request_id="r1"), "e1")
    assert observation.exit_code is None and "0 matches" in observation.content


def test_str_replace_multiple_ambiguous(tmp_path):
    executor = _executor(tmp_path)
    executor.write_text("b.txt", "aaa")
    observation = execute_action(executor,
                                 ToolAction("str_replace", {"path": "b.txt", "old": "a", "new": "b"}, request_id="r1"), "e1")
    assert observation.exit_code is None and "multiple matches" in observation.content


def test_str_replace_multiple_expected(tmp_path):
    executor = _executor(tmp_path)
    executor.write_text("b.txt", "aaa")
    observation = execute_action(executor,
                                 ToolAction("str_replace", {"path": "b.txt", "old": "a", "new": "b",
                                                            "expected_occurrences": 3}, request_id="r1"), "e1")
    assert observation.exit_code == 0 and executor.read_text("b.txt") == "bbb"
