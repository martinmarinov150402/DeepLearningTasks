import numpy as np

def mae(y_true, y_pred):
    errors = y_pred - y_true
    absolute_errors = np.abs(errors)
    return np.mean(absolute_errors)

def mse(y_true, y_pred):
    errors = y_pred - y_true
    squared_errors = errors ** 2
    return np.mean(squared_errors)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))