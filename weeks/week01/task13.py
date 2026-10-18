import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    figure, axes = plt.subplots(1, 3);

    rng = np.random.default_rng(seed=42)
    size_of_array = 10000

    uniform_array = np.array(rng.uniform(size = size_of_array));
    normal_array = np.array(rng.normal(size = size_of_array));
    exponential_array = np.array(rng.exponential(size = size_of_array));

    axes[0].hist(uniform_array, bins=30)
    axes[0].set_title('Uniform Distribution')
    axes[1].hist(normal_array, bins=30)
    axes[1].set_title('Normal Distribution')
    axes[2].hist(exponential_array, bins=30)
    axes[2].set_title('Exponential Distribution')
    plt.show()

