import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts" / "build_ast_bundles.py"
_SPEC = importlib.util.spec_from_file_location("build_ast_bundles", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _task(repo: str, kind: str, index: int) -> dict:
    return {
        "task_id": f"{repo}-{kind}-{index}",
        "repo": repo,
        "metadata": {"task_source": "AST_MUTATION"},
        "mutation": {"kind": kind},
    }


def test_select_ast_tasks_balances_repo_and_mutation_kind():
    tasks = [
        *[_task("alpha", "invert", i) for i in range(4)],
        *[_task("alpha", "constant", 10 + i) for i in range(4)],
        *[_task("beta", "invert", 20 + i) for i in range(4)],
    ]

    selected = _MODULE.select_ast_tasks(tasks, 6)

    assert len(selected) == 6
    assert [(task["repo"], task["mutation"]["kind"]) for task in selected] == [
        ("alpha", "constant"),
        ("alpha", "invert"),
        ("beta", "invert"),
        ("alpha", "constant"),
        ("alpha", "invert"),
        ("beta", "invert"),
    ]


def test_select_ast_tasks_rejects_shortage():
    try:
        _MODULE.select_ast_tasks([_task("alpha", "invert", 0)], 2)
    except ValueError as exc:
        assert "need 2" in str(exc) and "found 1" in str(exc)
    else:
        raise AssertionError("shortage must fail")

