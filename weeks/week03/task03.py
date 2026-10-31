import importlib
import dl_lib.perceptrons.datasets as datasets
import dl_lib.perceptrons.functional as functional
import dl_lib.losses.functional as losses

import numpy as np


def main():

    dataset = datasets.create_dataset(10)

    w = functional.initialize_weights(0, 10, 1)
    y_true = np.array([y for _, y in dataset])
    x = np.array([x for x, _ in dataset])
    x = x.reshape(-1, 1)
    weightLoss = lambda w: functional.dataset_loss(
        x, y_true, w, 0, activation=None
    )
    gradient = functional.numerical_gradient(weightLoss, w)

    print(f"Loss before update: {weightLoss(w)}")
    print(f"Numerical gradient: {gradient}")

    newW_raw = w - gradient

    print(
        f"w after subtracting the raw gradient: "
        f"w={newW_raw}, loss={weightLoss(newW_raw)}"
    )

    newW_lr = w - 0.001 * gradient

    print(
        f"w after one step with learning_rate=0.001: "
        f"w={newW_lr}, loss={weightLoss(newW_lr)}"
    )

    print(f"Loss before update: {weightLoss(w)}")

    for i in range(500):
        grad = functional.numerical_gradient(weightLoss, w)

        w = functional.gradient_descent_step(w, grad, 0.001)

        print(
            f"epoch {i + 1}: w={w}, loss={weightLoss(w)}"
        )