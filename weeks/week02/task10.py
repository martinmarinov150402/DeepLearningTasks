import importlib
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

def main():

    activations = importlib.import_module("dl_lib.activations.functional")

    print(f"sigmoid(-10) = {activations.sigmoid(-10):.6f}")
    print(f"sigmoid(0)   = {activations.sigmoid(0):.6f}")
    print(f"sigmoid(10)  = {activations.sigmoid(10):.6f}")

    x_values = np.arange(-10, 10, 0.1)

    sigmoids = [activations.sigmoid(x) for x in x_values]

    figure, axis = plt.subplots()


    axis.plot(x_values, sigmoids)

    plt.show()