import numpy as np

def main():
    y_true = np.array([3., 5., 7., 9.])
    y_pred = np.array([2., 6., 8., 7.])

    errors = y_pred - y_true
    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2
    mean_error = np.mean(errors)
    mean_absolute_error = np.mean(absolute_errors)
    mean_squared_error = np.mean(squared_errors)
    root_mean_squared_error = np.sqrt(mean_squared_error)

    print("y_true: ", y_true)
    print("y_pred: ", y_pred)
    print("errors: ", errors)
    print("absolute errors: ", absolute_errors)
    print("squared errors: ", squared_errors)
    print("")
    print("mean error: ", mean_error)
    print("MAE: ", mean_absolute_error)
    print("MSE: ", mean_squared_error)
    print("RMSE: ", root_mean_squared_error)