import importlib
import numpy as np

def main():
    losses = importlib.import_module("dl_lib.losses.functional")
    perceptrons = importlib.import_module("dl_lib.perceptrons.functional")

    dataset = perceptrons.create_dataset(10)

    w = perceptrons.initialize_weights(0,10,1)

    y_true = np.array([i[1] for i in dataset])
    x = np.array([i[0] for i in dataset])

    gradient = perceptrons.numerical_gradient(losses.mse,w,dataset)
    print(f"Loss before update: {losses.mse(y_true, np.dot(w, np.atleast_2d(x)))}")
    print(f"Numerical gradient: {gradient}")

    newW_raw = w - gradient
    print(f"w after subtracting the raw gradient: w= {newW_raw}, loss={losses.mse(y_true, np.dot(newW_raw, np.atleast_2d(x)))}")

    newW_lr = w - 0.001 * gradient
    print(f"w after one step with learning_rate=0.001: w={newW_lr}, loss={losses.mse(y_true, np.dot(newW_lr, np.atleast_2d(x)))}")

    print(f"Loss before update: {losses.mse(y_true, np.dot(w, np.atleast_2d(x)))}")

    for i in range(0, 10):
        grad = perceptrons.numerical_gradient(losses.mse,w,dataset)
        w = perceptrons.gradient_descent_step(w, grad, 0.001)

        print(f"epoch  {i+1}: w={w}, loss={losses.mse(y_true, np.dot(w, np.atleast_2d(x)))}")