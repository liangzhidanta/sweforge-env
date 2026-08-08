from sweforge.env_server.docker.executors import LocalExecutor
from sweforge.env_server.docker.tools import execute_action
from sweforge.protocol.tools import (
    BashAction,
    SearchAction,
    StrReplaceAction,
    ViewFileAction,
)


def _executor(tmp_path):
    (tmp_path / "a.py").write_text("def get_or_compute(key):\n    return key\n")
    return LocalExecutor(tmp_path)


def test_bash_success(tmp_path):
    observation = execute_action(_executor(tmp_path), BashAction(command="echo ok"))
    assert observation.exit_code == 0
    assert observation.stdout.strip() == "ok"


def test_search_found(tmp_path):
    observation = execute_action(_executor(tmp_path), SearchAction(query="get_or_compute"))
    assert len(observation.matches) == 1
    assert observation.matches[0].path == "a.py"
    assert observation.matches[0].line == 1
    assert "get_or_compute" in observation.matches[0].content


def test_search_no_match(tmp_path):
    observation = execute_action(_executor(tmp_path), SearchAction(query="zzz_no_match"))
    assert observation.matches == []
    assert observation.truncated is False


def test_search_truncated(tmp_path):
    executor = LocalExecutor(tmp_path)
    executor.write_text("b.txt", "\n".join(f"needle {i}" for i in range(250)))
    observation = execute_action(executor, SearchAction(query="needle"))
    assert observation.truncated is True
    assert len(observation.matches) == 200  # _MAX_SEARCH_MATCHES


def test_search_invalid_regex(tmp_path):
    observation = execute_action(_executor(tmp_path), SearchAction(query="[unclosed"))
    assert observation.matches == []
    assert observation.error is not None


def test_search_path_not_found(tmp_path):
    observation = execute_action(_executor(tmp_path), SearchAction(query="x", path="missing.txt"))
    assert observation.matches == []
    assert "path not found" in observation.error


def test_search_escape(tmp_path):
    observation = execute_action(_executor(tmp_path), SearchAction(query="x", path="../secret.txt"))
    assert observation.matches == []
    assert observation.error is not None


def test_view_file_range(tmp_path):
    observation = execute_action(
        _executor(tmp_path), ViewFileAction(path="a.py", start_line=1, end_line=2)
    )
    assert observation.start_line == 1 and observation.end_line == 2
    assert "def get_or_compute" in observation.content
    assert observation.total_lines == 2


def test_view_file_default_bounds(tmp_path):
    observation = execute_action(_executor(tmp_path), ViewFileAction(path="a.py"))
    assert observation.start_line == 1 and observation.end_line == 2
    assert observation.content.splitlines()[0].startswith("     1\t")


def test_view_file_not_found(tmp_path):
    observation = execute_action(_executor(tmp_path), ViewFileAction(path="missing.txt"))
    assert observation.error is not None and observation.start_line == 0


def test_view_file_is_directory(tmp_path):
    observation = execute_action(_executor(tmp_path), ViewFileAction(path="."))
    assert observation.error is not None and "directory" in observation.error


def test_view_file_out_of_bounds(tmp_path):
    observation = execute_action(
        _executor(tmp_path), ViewFileAction(path="a.py", start_line=5, end_line=9)
    )
    assert observation.error is not None and "out of bounds" in observation.error


def test_view_file_escape(tmp_path):
    observation = execute_action(_executor(tmp_path), ViewFileAction(path="../secret.txt"))
    assert observation.error is not None and observation.start_line == 0


def test_str_replace_single(tmp_path):
    executor = _executor(tmp_path)
    observation = execute_action(
        executor,
        StrReplaceAction(path="a.py", old_string="return key", new_string="return key.upper()"),
    )
    assert observation.success is True
    assert executor.read_text("a.py") == "def get_or_compute(key):\n    return key.upper()\n"


def test_str_replace_zero(tmp_path):
    observation = execute_action(
        _executor(tmp_path),
        StrReplaceAction(path="a.py", old_string="absent", new_string="x"),
    )
    assert observation.success is False
    assert "not found" in observation.error


def test_str_replace_not_unique(tmp_path):
    executor = _executor(tmp_path)
    executor.write_text("b.txt", "aaa")
    observation = execute_action(
        executor, StrReplaceAction(path="b.txt", old_string="a", new_string="b")
    )
    assert observation.success is False
    assert "not unique" in observation.error


def test_str_replace_empty_old_inserts_at_head(tmp_path):
    executor = _executor(tmp_path)
    observation = execute_action(
        executor, StrReplaceAction(path="a.py", old_string="", new_string="# header\n")
    )
    assert observation.success is True
    assert executor.read_text("a.py").startswith("# header\n")


def test_str_replace_empty_old_creates_file(tmp_path):
    executor = _executor(tmp_path)
    observation = execute_action(
        executor, StrReplaceAction(path="new.txt", old_string="", new_string="hello")
    )
    assert observation.success is True
    assert executor.read_text("new.txt") == "hello"


def test_str_replace_escape(tmp_path):
    observation = execute_action(
        _executor(tmp_path), StrReplaceAction(path="../secret.txt", old_string="x", new_string="y")
    )
    assert observation.success is False
    assert observation.error is not None
