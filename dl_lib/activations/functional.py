import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return int(x > 0) * x

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def naive_softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def softmax(z):
    m = np.max(z)
    exp_z = np.exp(z - m)
    return exp_z / np.sum(exp_z)

