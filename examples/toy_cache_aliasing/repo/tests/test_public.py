import unittest

from toy_cache.cache import get_or_compute


class PublicTests(unittest.TestCase):
    def test_correct_value_returned(self):
        self.assertEqual(get_or_compute("k", lambda: 42), 42)

    def test_repeat_returns_same_value(self):
        self.assertEqual(get_or_compute("r", lambda: 7), 7)
        self.assertEqual(get_or_compute("r", lambda: 7), 7)
