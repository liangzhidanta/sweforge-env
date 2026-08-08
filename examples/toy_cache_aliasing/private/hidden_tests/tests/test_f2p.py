import unittest

from toy_cache.cache import get_or_compute


class F2PTest(unittest.TestCase):
    def test_repeated_key_does_not_recompute(self):
        calls: list[str] = []

        def compute() -> int:
            calls.append("compute")
            return 42

        self.assertEqual(get_or_compute("k", compute), 42)
        self.assertEqual(get_or_compute("k", compute), 42)
        self.assertEqual(calls, ["compute"])
