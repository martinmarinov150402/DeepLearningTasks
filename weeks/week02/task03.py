import numpy as np
import importlib

def main():
    y_true = np.array([3., 5., 7., 9.])
    y_pred = np.array([2., 6., 8., 7.])
    
    functional = importlib.import_module('dl_lib.losses.functional')
    print("y_true: ", y_true)
    print("y_pred: ", y_pred)
    print("MSE: ", functional.mse(y_true, y_pred))

    y_pred[3] = 100.0
    print("y pred with an extreme outlier: ", y_pred)
    print("MAE with outlier: ", functional.mae(y_true, y_pred))
    print("MSE with outlier: ", functional.mse(y_true, y_pred))