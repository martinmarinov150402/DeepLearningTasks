import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def main():
    size_of_array = 50
    rng = np.random.default_rng(seed=42)
    x = np.array(rng.uniform(1, 100, size_of_array));

    noise = np.array(rng.normal(-10, 10, size_of_array));

    slope = 2
    intercept = 1;

    y = np.multiply(x, slope);
    y = np.add(y, intercept);
    y = np.add(y, noise);

    figure, axis = plt.subplots()
    axis.scatter(x, y)

    plt.show()




