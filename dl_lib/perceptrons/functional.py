import numpy as np

def create_dataset(n):
    return [(x, x * 2) for x in range(n)]


def initialize_weights(lower_bound, upper_bound, n, seed = 42):

    rng = np.random.default_rng(seed);

    return rng.uniform(lower_bound, upper_bound, n)