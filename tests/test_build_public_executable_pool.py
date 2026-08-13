import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts" / "build_public_executable_pool.py"
_SPEC = importlib.util.spec_from_file_location("build_public_executable_pool", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


PATCH = """--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1,2 +1,2 @@
-VALUE = 0
+VALUE = 1
 x = 2

--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -3,5 +3,6 @@ class TestCore:
     def test_value(self):
         value = core.value()
+        assert value == 1
         assert value

     def test_other(self):
"""


def test_test_only_patch_keeps_only_test_file_diffs():
    selected = _MODULE.test_only_patch(PATCH)

    assert "tests/test_core.py" in selected
    assert "pkg/core.py" not in selected


def test_source_only_patch_removes_hidden_test_diffs():
    selected = _MODULE.source_only_patch(PATCH)

    assert "pkg/core.py" in selected
    assert "tests/test_core.py" not in selected


def test_changed_test_selectors_maps_added_line_to_enclosing_method():
    patched = """from pkg import core

class TestCore:
    def test_value(self):
        value = core.value()
        assert value == 1
        assert value

    def test_other(self):
        assert True
"""

    selectors = _MODULE.changed_test_selectors(
        {"tests/test_core.py": patched},
        PATCH,
    )

    assert selectors == ["tests/test_core.py::TestCore::test_value"]


def test_finalize_task_records_real_runtime_and_verified_tests():
    task = {
        "task_id": "r2e-demo-1",
        "repo": "demo",
        "environment": {"image": "org/demo:1"},
        "gold_patch": PATCH,
        "fail_to_pass": [{"test_id": "wrong", "kind": "fail_to_pass"}],
        "pass_to_pass": [],
        "metadata": {"task_source": "PUBLIC_EXECUTABLE"},
    }

    result = _MODULE.finalize_task(
        task,
        f2p=["tests/test_core.py::TestCore::test_value"],
        p2p=["tests/test_core.py::TestCore::test_other"],
        evidence={"base_f2p_rc": 1, "fix_f2p_rc": 0},
    )

    assert result["environment"]["workspace"] == "/testbed"
    assert result["environment"]["runtime_user"] == "0:0"
    assert result["environment"]["seed_from_snapshot"] is False
    assert result["fail_to_pass"][0]["test_id"].endswith("test_value")
    assert result["metadata"]["executable_verified"] is True
    assert "tests/test_core.py" not in result["gold_patch"]
