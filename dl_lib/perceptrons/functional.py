import numpy as np
import dl_lib.losses.functional as losses

def initialize_weights(lower_bound, upper_bound, size = 1, seed=42):

    rng = np.random.default_rng(seed)
    return rng.uniform(lower_bound, upper_bound, size)

def dataset_loss(dataset, targets, weights, bias=0, activation=None, loss = losses.mse):
    
    dataset = np.atleast_2d(dataset)
    targets = np.atleast_2d(targets)
    weights = np.atleast_2d(weights)
    
    Z = (weights @ dataset.T) + bias
  

    if activation is not None:
        y_pred = activation(Z)
    else:
        y_pred = Z 
    
    return loss(targets, y_pred)



def dataset_loss_given_bias(dataset, targets, weights, activation=None, bias=[0], loss = losses.mse ):
    
    dataset = np.array(dataset)
    targets = np.array(targets)
    
    Z = np.add(np.dot(dataset, weights), bias)
  

    if activation is not None:
        y_pred = activation(Z)
    else:
        y_pred = Z 
    
    return loss(targets, y_pred)


          


def numerical_gradient(loss_function, w):
    epsilon = 1e-5

    grad = np.zeros_like(w, dtype=float)

    for index in np.ndindex(w.shape):
        e_i = np.zeros_like(w, dtype=float)
        e_i[index] = 1.0

        loss_plus = loss_function(w + epsilon * e_i)
        loss_minus = loss_function(w - epsilon * e_i)

        grad[index] = (loss_plus - loss_minus) / (2 * epsilon)

    return grad



def analytical_gradient(dataset, w, bias=0, activation=None, activation_derivative=None, loss="mse"):

    x = np.array([x for x, _ in dataset])
    y = np.array([y for _, y in dataset])

    N = len(y)

    z = x @ w + bias

    if activation is not None:
        y_pred = activation(z)
    else:
        y_pred = z

    match loss:
        case "mse":
            error = y_pred - y

            if activation is not None and activation_derivative is not None:
                error *= activation_derivative(z)

            grad_w = (2 / N) * (x.T @ error)
            grad_b = (2 / N) * np.sum(error)

        case "bce":
            error = y_pred - y

            grad_w = (1 / N) * (x.T @ error)
            grad_b = (1 / N) * np.sum(error)

        case _:
            raise ValueError("Unknown loss function")

    return grad_w, grad_b



def gradient_descent_step(w, gradient, learning_rate):
    return w - learning_rate * gradient

def forward(x, weights, bias = 0, activation = None):
        data = np.sum(np.dot(weights, x)) + bias
        if activation != None:
            return activation(data)
        else:
            return data
