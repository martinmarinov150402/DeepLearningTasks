import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return (x > 0) * x


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def naive_softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def softmax(z, axis = -1, keepdims = True):
    m = np.max(z)
    exp_z = np.exp(z - m)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=keepdims)
