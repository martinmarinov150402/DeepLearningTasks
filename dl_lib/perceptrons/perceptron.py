import importlib

import numpy as np
import dl_lib.perceptrons.functional as functional


class Perceptron:
    activation = (None,)
    bias = 0
    use_bias = (False,)

    def __init__(
        self, low, high, numberOfArgs, use_bias=False, activation=None, seed=42
    ):
        functional = importlib.import_module("dl_lib.perceptrons.functional")
        self.weights = functional.initialize_weights(low, high, numberOfArgs, seed)
        self.activation = activation
        self.use_bias = use_bias


    def forward(self, n_inputs):
        return functional.forward(n_inputs, self.weights, self.bias, self.activation)
