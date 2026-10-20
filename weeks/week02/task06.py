import numpy as np
import importlib

def main():
    functional = importlib.import_module("dl_lib.metrics.functional");

    y_true = np.array([1, 0, 1, 0, 1])
    y_pred =  np.array([1, 0, 0, 1, 1])

    confusion_matrix = functional.confusion_matrix(y_pred, y_true);

    print(f"True positives: {confusion_matrix["tp"]}")
    print(f"True negatives: {confusion_matrix["tn"]}")
    print(f"False positives: {confusion_matrix["fp"]}")
    print(f"False negatives: {confusion_matrix["fn"]}")

