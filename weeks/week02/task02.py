import numpy as np
import importlib

def main():
    y_true = np.array([3., 5., 7., 9.])
    y_pred = np.array([2., 6., 8., 7.])

    print("y_true: ", y_true)
    print("y_pred: ", y_pred)

    functional = importlib.import_module('dl_lib.losses.functional')
    print("MAE: ", functional.mae(y_true, y_pred))