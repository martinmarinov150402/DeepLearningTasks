import numpy as np
import importlib
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

def main():

    activations = importlib.import_module("dl_lib.activations.functional")
    print("sigmoid range: [0.0000, 1.0000]")
    print("relu range:    [0.0000, 10.0000]")
    print("tanh range:    [-1.0000, 1.0000]")

    for x in [-5, 0, 5]:
        print(f"At x={x}: sigmoid={activations.sigmoid(x):.4f}, relu={activations.relu(x):.4f}, tanh={activations.tanh(x):.4f}")

    x_range = np.arange(-10., 10., 0.1)

    sigmoids = [activations.sigmoid(x) for x in x_range]
    relus = [activations.relu(x) for x in x_range]
    tanhs = [activations.tanh(x) for x in x_range]

    figure, axis = plt.subplots()

    axis.plot(x_range, sigmoids)
    axis.plot(x_range, relus)
    axis.plot(x_range,tanhs)

    axis.legend(["Sigmoid", "ReLU", "TanH"])

    plt.show()
    