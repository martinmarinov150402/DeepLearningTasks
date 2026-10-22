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

def binary_cross_entropy(p, y):
    N = np.size(p)

    bce = -1/N * np.sum(y * np.log(p) + (1 - y)*np.log(1-p))

    return bce;

def categorical_cross_entropy(y_true, p):

    N = np.size(y_true)
    num_classes = np.size(p[0])
    new_true = np.array([
        [int(j == y_true[i]) for j in range (0, num_classes)] for i in range(0,N) 
    ])
    
    cce = -1/N * np.sum(new_true * np.log(p))
    return cce
  
