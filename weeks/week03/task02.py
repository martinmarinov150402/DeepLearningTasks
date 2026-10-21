import numpy as np
import importlib

def main():
    functional = importlib.import_module("dl_lib.perceptrons.functional")

    errors_functional = importlib.import_module("dl_lib.losses.functional")

    dataset = functional.create_dataset(6)

    weights = functional.initialize_weights(0,10,1,42)



    print(dataset)

    print(weights)

    x = [z[0] for z in dataset]

    y_true = [z[1] for z in dataset]
    y_pred = np.dot(weights[0], x);

    print("MSE: ", errors_functional.mse(y_true, y_pred))

    print("Perturbing w:")
    for i in [0.002, 0.001, -0.001, -0.002]:
        print(f"w + ({i}) -> {weights[0] + i}, MSE={errors_functional.mse(y_true, np.dot(weights[0] + i, x))}")
    

    