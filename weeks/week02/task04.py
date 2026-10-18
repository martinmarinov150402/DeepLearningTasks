import numpy as np
import matplotlib
import importlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def main():

    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([4, -1.5, 2.5, 7])

    figure, axis = plt.subplots()

    mae_values = []
    mse_values = []
    rmse_values = []

    functional = importlib.import_module("dl_lib.losses.functional")

    for i in range(0, 11):
        mae_values.append(functional.mae(y_true, y_pred))
        mse_values.append(functional.mse(y_true, y_pred))
        rmse_values.append(functional.rmse(y_true, y_pred))

        y_pred[3] += 1

    axis.plot(range(0, 11), mae_values, label='MAE', color='blue')
    axis.plot(range(0, 11), mse_values, label='MSE', color='orange')
    axis.plot(range(0, 11), rmse_values, label='RMSE', color='green')
    axis.legend(["MAE", "MSE","RMSE"])

    plt.show()