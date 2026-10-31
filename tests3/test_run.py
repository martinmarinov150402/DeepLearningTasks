import importlib
import unittest


class TestRun(unittest.TestCase):
    def test_run(self):
        importlib.import_module("run")
