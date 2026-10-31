import importlib

import numpy as np


def main():
    y_true = np.array([3.0, 5.0, 7.0, 9.0])
    y_pred = np.array([2.0, 6.0, 8.0, 7.0])

    print("y_true: ", y_true)
    print("y_pred: ", y_pred)

    functional = importlib.import_module("dl_lib.losses.functional")
    print("MAE: ", functional.mae(y_true, y_pred))
