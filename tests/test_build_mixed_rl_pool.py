import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "build_mixed_rl_pool.py"
_SPEC = importlib.util.spec_from_file_location("build_mixed_rl_pool", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _task(source: str, repo: str, index: int, kind: str = "kind") -> dict:
    return {
        "task_id": f"{source.lower()}-{repo}-{index:03d}",
        "repo": repo,
        "base_commit": f"{index:040x}",
        "problem_statement": "fix the bug",
        "environment": {
            "image": None,
            "setup_commands": [],
            "build_commands": [],
            "test_commands": {"t": ["python", "-m", "pytest", "-q", "t"]},
        },
        "fail_to_pass": [{"test_id": "t", "kind": "fail_to_pass", "timeout_seconds": 300}],
        "pass_to_pass": [],
        "gold_patch": "diff --git a/a.py b/a.py\n",
        "mutation": {"kind": kind},
        "difficulty": None,
        "metadata": {
            "task_source": source,
            "repo_id": repo,
            "generator_type": kind,
            "generator_version": "v1",
        },
    }


def test_select_quota_is_exact_and_round_robin_across_repos():
    tasks = [
        *[_task("PUBLIC_EXECUTABLE", "alpha", i) for i in range(4)],
        *[_task("PUBLIC_EXECUTABLE", "beta", 10 + i) for i in range(4)],
    ]

    selected = _MODULE.select_quota(tasks, "PUBLIC_EXECUTABLE", 5)

    assert len(selected) == 5
    assert [task["repo"] for task in selected] == ["alpha", "beta", "alpha", "beta", "alpha"]


def test_select_quota_rejects_shortage_and_wrong_source():
    tasks = [_task("AST_MUTATION", "alpha", 0)]

    with pytest.raises(ValueError, match="PUBLIC_EXECUTABLE.*need 1.*found 0"):
        _MODULE.select_quota(tasks, "PUBLIC_EXECUTABLE", 1)


def test_build_mixed_pool_enforces_100_50_50_and_redacts_pool_secrets():
    public = [_task("PUBLIC_EXECUTABLE", "pub", i) for i in range(101)]
    ast = [_task("AST_MUTATION", "ast", 200 + i, "invert_condition") for i in range(51)]
    reversal = [_task("REPAIR_REVERSAL", "rev", 300 + i, "real_repair_reversal") for i in range(51)]

    full, pool, manifest = _MODULE.build_mixed_pool(
        public,
        ast,
        reversal,
        quotas={"PUBLIC_EXECUTABLE": 100, "AST_MUTATION": 50, "REPAIR_REVERSAL": 50},
    )

    assert len(full) == len(pool) == 200
    assert manifest["source_counts"] == {
        "AST_MUTATION": 50,
        "PUBLIC_EXECUTABLE": 100,
        "REPAIR_REVERSAL": 50,
    }
    assert len({task["task_id"] for task in full}) == 200
    assert all(task["gold_patch"] is None and task["mutation"] is None for task in pool)


def test_build_mixed_pool_rejects_duplicate_ids():
    duplicate = _task("PUBLIC_EXECUTABLE", "pub", 0)
    public = [duplicate, *[_task("PUBLIC_EXECUTABLE", "pub", i) for i in range(1, 100)]]
    ast = [_task("AST_MUTATION", "ast", 200 + i) for i in range(50)]
    reversal = [_task("REPAIR_REVERSAL", "rev", 300 + i) for i in range(50)]
    reversal[0]["task_id"] = duplicate["task_id"]

    with pytest.raises(ValueError, match="duplicate task_id"):
        _MODULE.build_mixed_pool(
            public,
            ast,
            reversal,
            quotas={"PUBLIC_EXECUTABLE": 100, "AST_MUTATION": 50, "REPAIR_REVERSAL": 50},
        )

