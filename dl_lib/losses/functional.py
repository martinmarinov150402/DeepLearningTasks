import numpy as np


def mae(y_true, y_pred):
    errors = y_pred - y_true
    absolute_errors = np.abs(errors)
    return np.mean(absolute_errors)


def mse(y_true, y_pred):
    errors = y_pred - y_true
    squared_errors = errors**2
    return np.mean(squared_errors)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def binary_cross_entropy(y, p):
    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if p.shape != y.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    epsilon = 1e-12
    p = np.clip(p, epsilon, 1 - epsilon)

    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def categorical_cross_entropy(y_true, p):

    N = np.size(y_true)
    num_classes = np.size(p[0])
    new_true = np.array(
        [[int(j == y_true[i]) for j in range(num_classes)] for i in range(N)]
    )

    cce = -1 / N * np.sum(new_true * np.log(p))
    return cce
