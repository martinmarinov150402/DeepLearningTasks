import importlib
import numpy as np

import dl_lib.losses.functional as losses
import dl_lib.perceptrons.functional as functional


def train(
    perceptron,
    dataset,
    learning_rate,
    epochs,
    loss_function = losses.mse,
    record_freq=1,
):

    history = []
    y_true = np.array([y for _, y in dataset])
    x = np.array([x for x, _ in dataset])
    weightLoss = lambda w: functional.dataset_loss(
        x, y_true, w, perceptron.bias, activation=perceptron.activation, loss=loss_function
    )

    bias_loss = lambda b: functional.dataset_loss_given_bias(
                x, y_true, perceptron.weights, perceptron.activation, np.array([b])
            )
            

    for i in range(epochs):
        grad = functional.numerical_gradient(weightLoss, perceptron.weights)
        #print(grad)

        perceptron.weights = functional.gradient_descent_step(perceptron.weights, grad, learning_rate)
        if(perceptron.use_bias):
            bias_gradient = functional.numerical_gradient(bias_loss, np.array([perceptron.bias]))[0]
            perceptron.bias = functional.gradient_descent_step(perceptron.bias, bias_gradient, learning_rate)


        loss = weightLoss(perceptron.weights)
        if (i+1) % record_freq == 0:
            history.append((i+1, loss))
        #print(
        #    f"epoch {i + 1}: w={perceptron.weights}, bias={perceptron.bias}, loss={loss}"
        #)

    return history


