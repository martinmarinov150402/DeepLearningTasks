import numpy as np
import importlib

def generate_dataset(n=1000, positive_rate=0.5, accuracy=0.9, seed=None):
    rng = np.random.default_rng(seed)

    y_true = rng.choice(
        [0, 1],
        size=n,
        p=[1 - positive_rate, positive_rate]
    )

    y_pred = y_true.copy()

    errors = rng.random(n) > accuracy
    y_pred[errors] = 1 - y_pred[errors]

    return y_pred, y_true


def main():

    functional = importlib.import_module("dl_lib.metrics.functional")
    print("Balanced dataset:")
    y_pred_balanced, y_true_balanced = generate_dataset(
        n=1000,
        positive_rate=0.5,
        accuracy=0.6
    )

    print("accuracy: ", functional.accuracy(y_pred_balanced, y_true_balanced))
    print("precision: ", functional.precision(y_pred_balanced, y_true_balanced))
    print("recall: ", functional.recall(y_pred_balanced, y_true_balanced))
    print("f1: ", functional.f1_score(y_pred_balanced, y_true_balanced))

    print("Imbalanced dataset (990 negative, 10 positive):")

    y_pred_imbalanced, y_true_imbalanced = y_pred, y_true = generate_dataset(
        n=1000,
        positive_rate=0.01,
        accuracy=0.99
    )

    print("accuracy: ", functional.accuracy(y_pred_imbalanced, y_true_imbalanced))
    print("precision: ", functional.precision(y_pred_imbalanced, y_true_imbalanced))
    print("recall: ", functional.recall(y_pred_imbalanced, y_true_imbalanced))
    print("f1: ", functional.f1_score(y_pred_imbalanced, y_true_imbalanced))

    

    