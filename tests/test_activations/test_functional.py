import unittest

import numpy as np
import torch

from dl_lib.activations import functional


class TestSigmoid(unittest.TestCase):

    def test_when_input_is_zero_then_returns_one_half(self):
        
        x = np.array([0.0])
        expected = 0.5

        
        actual = functional.sigmoid(x)[0]

        
        self.assertAlmostEqual(actual, expected)

    def test_when_input_is_very_negative_then_approaches_zero(self):
        
        x = np.array([-100.0])

        
        actual = functional.sigmoid(x)[0]

        
        self.assertAlmostEqual(actual, 0.0)

    def test_when_input_is_very_positive_then_approaches_one(self):
        
        x = np.array([100.0])

        
        actual = functional.sigmoid(x)[0]

        
        self.assertAlmostEqual(actual, 1.0)

    def test_when_called_then_matches_torch_sigmoid(self):
        
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        expected = torch.sigmoid(torch.from_numpy(x)).numpy()

        
        actual = functional.sigmoid(x)

        
        np.testing.assert_allclose(actual, expected)


class TestSigmoidDerivative(unittest.TestCase):

    def test_when_input_is_zero_then_returns_one_quarter(self):
        
        x = np.array([0.0])
        expected = 0.25

        
        actual = functional.sigmoid_derivative(x)[0]

        
        self.assertAlmostEqual(actual, expected)

    def test_when_input_is_very_negative_or_positive_then_approaches_zero(self):
        
        x = np.array([-100.0, 100.0])
        expected = np.array([0.0, 0.0])

        
        actual = functional.sigmoid_derivative(x)

        
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_when_called_then_matches_the_numerical_derivative_of_sigmoid(self):
        
        x = np.array([-2.0, -0.5, 0.3, 1.5])
        h = 1e-5
        expected = (functional.sigmoid(x + h) - functional.sigmoid(x - h)) / (2 * h)

        
        actual = functional.sigmoid_derivative(x)

        
        np.testing.assert_allclose(actual, expected, atol=1e-6)


class TestRelu(unittest.TestCase):

    def test_when_input_is_negative_then_returns_zero(self):
        
        x = np.array([-5.0, -1.0])
        expected = np.array([0.0, 0.0])

        
        actual = functional.relu(x)

        
        np.testing.assert_allclose(actual, expected)

    def test_when_input_is_positive_then_returns_input_unchanged(self):
        
        x = np.array([2.0, 7.5])

        
        actual = functional.relu(x)

        
        np.testing.assert_allclose(actual, x)

    def test_when_called_then_matches_torch_relu(self):
        
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        expected = torch.nn.functional.relu(torch.from_numpy(x)).numpy()

        
        actual = functional.relu(x)

        
        np.testing.assert_allclose(actual, expected)


class TestTanh(unittest.TestCase):

    def test_when_input_is_zero_then_returns_zero(self):
        
        x = np.array([0.0])
        expected = 0.0

        
        actual = functional.tanh(x)[0]

        
        self.assertAlmostEqual(actual, expected)

    def test_when_input_is_very_negative_then_approaches_negative_one(self):
        
        x = np.array([-100.0])

        
        actual = functional.tanh(x)[0]

        
        self.assertAlmostEqual(actual, -1.0)

    def test_when_called_then_matches_torch_tanh(self):
        
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        expected = torch.tanh(torch.from_numpy(x)).numpy()

        
        actual = functional.tanh(x)

        
        np.testing.assert_allclose(actual, expected)


class TestSoftmax(unittest.TestCase):

    def test_when_called_then_probabilities_sum_to_one(self):
        
        x = np.array([1.0, 2.0, 3.0])
        expected = 1.0

        
        actual = functional.softmax(x).sum()

        
        self.assertAlmostEqual(actual, expected)

    def test_when_input_has_large_values_then_does_not_overflow(self):
        
        x = np.array([1000.0, 1001.0, 1002.0])

        
        actual = functional.softmax(x)

        
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertAlmostEqual(actual.sum(), 1.0)

    def test_when_called_then_matches_torch_softmax(self):
        
        x = np.array([1.0, 2.0, 3.0])
        expected = torch.softmax(torch.from_numpy(x), dim=-1).numpy()

        
        actual = functional.softmax(x)

        
        np.testing.assert_allclose(actual, expected)

    def test_when_input_is_batched_then_normalizes_each_row_independently(self):
        
        x = np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]])
        expected_row_sums = np.array([1.0, 1.0])

        
        actual = functional.softmax(x)

        
        np.testing.assert_allclose(actual.sum(axis=-1), expected_row_sums)
