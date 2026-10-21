import importlib
import numpy as np

import matplotlib

matplotlib.use("TkAgg");

import matplotlib.pyplot as plt

def generate_dataset(n_samples=1000, seed=42):
    rng = np.random.default_rng(seed)

    y_true = rng.integers(0, 2, size=n_samples)

    y_pred = np.empty(n_samples)

    positive_indices = y_true == 1
    negative_indices = y_true == 0

    y_pred[positive_indices] = rng.normal(
        loc=0.75, scale=0.15, size=np.sum(positive_indices)
    )
    y_pred[negative_indices] = rng.normal(
        loc=0.25, scale=0.15, size=np.sum(negative_indices)
    )

    y_pred = np.clip(y_pred, 0, 1)
    return y_pred, y_true

def main():
    rng = np.random.default_rng(42);

    functional = importlib.import_module("dl_lib.metrics.functional");
    dataset_size = 1000
    y_pred, y_true = generate_dataset(dataset_size)

    thresholds = np.arange(0.05, 0.951, 0.05)

    precision = []
    recall = []

    for threshold in thresholds:
        current_pred = functional.apply_threshold(y_pred, threshold)

        precision.append(functional.precision(current_pred, y_true))
        recall.append(functional.recall(current_pred, y_true))

        print("y_true:", np.unique(y_true, return_counts=True))
        print("y_pred range:", np.min(y_pred), np.max(y_pred))
        print("\nThreshold | TP | FP | FN | Precision | Recall")

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        thresholds,
        precision,
        marker="o",
        label="precision"
    )

    axis.plot(
        thresholds,
        recall,
        marker="o",
        label="recall"
    )

    axis.set_title("Precision and recall as the classification threshold rises")
    axis.set_xlabel("threshold")
    axis.set_ylabel("value")

    axis.legend()
    axis.grid(True)

    plt.show()


    