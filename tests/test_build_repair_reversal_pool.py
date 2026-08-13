import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts" / "build_repair_reversal_pool.py"
_SPEC = importlib.util.spec_from_file_location("build_repair_reversal_pool", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


PATCH = """--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1 +1 @@
-VALUE = 0
+VALUE = 1

--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1,2 +1,3 @@
 def test_value():
+    assert value() == 1
     assert value()
"""


def test_finalize_reversal_has_canonical_source_and_source_only_gold():
    task = _MODULE.finalize_reversal(
        repo="demo",
        parent="a" * 40,
        fix_commit="b" * 40,
        subject="fix value",
        patch=PATCH,
        image="sweforge-repair:py311",
        f2p=["tests/test_core.py::test_value"],
        p2p=["tests/test_core.py::test_other"],
        evidence={"base_f2p_rc": 1, "fix_f2p_rc": 0},
    )

    assert task["metadata"]["task_source"] == "REPAIR_REVERSAL"
    assert task["mutation"]["kind"] == "real_repair_reversal"
    assert task["mutation"]["source_commit"] == "b" * 40
    assert "pkg/core.py" in task["gold_patch"]
    assert "tests/test_core.py" not in task["gold_patch"]
    assert task["environment"]["seed_from_snapshot"] is True


def test_fix_like_prioritizes_bug_fix_messages():
    assert _MODULE.fix_like("Fix incorrect timezone handling")
    assert _MODULE.fix_like("regression: restore iterator")
    assert not _MODULE.fix_like("Update documentation")

