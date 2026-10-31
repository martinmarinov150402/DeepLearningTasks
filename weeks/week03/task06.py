from dl_lib.losses.functional import binary_cross_entropy
from dl_lib.perceptrons.datasets import and_dataset
from dl_lib.perceptrons.perceptron import Perceptron
from dl_lib.perceptrons.training import train
import dl_lib.activations.functional as activations


def main():

    and_perceptron = Perceptron(-1, 1, 2, True, activations.sigmoid)

    train(and_perceptron, and_dataset(), 0.01, 100000, binary_cross_entropy, 1)

    print("Weights: ", and_perceptron.weights)
    print("Bias: ", and_perceptron.bias)

    for i in and_dataset():
        print(i[0], " ", and_perceptron.forward(i[0]))
