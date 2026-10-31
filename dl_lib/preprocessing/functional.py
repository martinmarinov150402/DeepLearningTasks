import numpy as np


def min_max_normalize(X, a = 0, b = 1):
    X_min = np.min(X)
    X_max = np.max(X)
    if(X_max - X_min == 0):
        return 0.
    X_norm = (X - X_min) / (X_max - X_min) * (b - a) + a
    return X_norm


def standardize(X):
    X_mean = np.mean(X)
    X_std = np.std(X)
    if(X_std == 0):
        return 0.
    X_standardized = (X - X_mean) / X_std
    return X_standardized
