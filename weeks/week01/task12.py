import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def main():
    figure, axes = plt.subplots(1, 2);

    sinX = np.arange(0, 6, 0.001)
    sinY = np.sin(sinX);

    cosX = np.arange(0, 6, 0.001)
    cosY = np.cos(cosX);

    axes[0].plot(sinX, sinY, label='sin(x)')
    axes[0].plot(cosX, cosY, label='cos(x)')
    axes[0].legend()
    
    rng = np.random.default_rng(seed=42)
    size_of_normal_array = 1000

    normal_array = np.array(rng.normal(size = size_of_normal_array));

    axes[1].hist(normal_array, bins=30)
    plt.show()
