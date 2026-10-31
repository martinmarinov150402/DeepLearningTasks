import unittest

import numpy as np

from dl_lib.perceptrons import datasets


class TestCreateDataset(unittest.TestCase):
    def test_when_called_then_returns_n_consecutive_x_and_2x_pairs(self):
        
        n = 4
        expected = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]

        
        actual = datasets.create_dataset(n)

        
        self.assertEqual(actual, expected)


class TestAndDataset(unittest.TestCase):
    def test_when_called_then_returns_the_and_truth_table(self):
        
        expected_targets = [0.0, 0.0, 0.0, 1.0]

        
        actual = datasets.and_dataset()

        
        self.assertEqual([y for _, y in actual], expected_targets)
        np.testing.assert_array_equal(
            np.array([x for x, _ in actual]),
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
        )


class TestOrDataset(unittest.TestCase):
    def test_when_called_then_returns_the_or_truth_table(self):
        
        expected_targets = [0.0, 1.0, 1.0, 1.0]

        
        actual = datasets.or_dataset()

        
        self.assertEqual([y for _, y in actual], expected_targets)
        np.testing.assert_array_equal(
            np.array([x for x, _ in actual]),
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
        )


class TestNandDataset(unittest.TestCase):
    def test_when_called_then_returns_the_nand_truth_table(self):
        
        expected_targets = [1.0, 1.0, 1.0, 0.0]

        
        actual = datasets.nand_dataset()

        
        self.assertEqual([y for _, y in actual], expected_targets)
        np.testing.assert_array_equal(
            np.array([x for x, _ in actual]),
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
        )


class TestXorDataset(unittest.TestCase):
    def test_when_called_then_returns_the_xor_truth_table(self):
        
        expected_targets = [0.0, 1.0, 1.0, 0.0]

        
        actual = datasets.xor_dataset()

        
        self.assertEqual([y for _, y in actual], expected_targets)
        np.testing.assert_array_equal(
            np.array([x for x, _ in actual]),
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
        )
