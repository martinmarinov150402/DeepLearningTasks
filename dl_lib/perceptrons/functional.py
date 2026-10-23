import numpy as np

def create_dataset(n):
    return [(x, x * 2) for x in range(n)]


def initialize_weights(lower_bound, upper_bound, n, seed = 42):

    rng = np.random.default_rng(seed);

    return rng.uniform(lower_bound, upper_bound, n)

def numerical_gradient(loss_function, w, dataset):
    epsilon = 0.001

    x = [i[0] for i in dataset]
    y_true = [i[1] for i in dataset]

    return (loss_function(y_true, np.dot(w + epsilon, np.atleast_2d(x))) - loss_function(y_true, np.dot(w, np.atleast_2d(x)))) / epsilon

def gradient_descent_step(w, gradient, learning_rate):
    return w - learning_rate * gradient

    

    
