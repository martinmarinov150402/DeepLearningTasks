import numpy as np
import importlib

def main():
    functional = importlib.import_module("dl_lib.metrics.functional")

    y_true = np.concatenate((np.zeros(990), np.ones(10))).astype(int)
    y_pred = np.zeros(1000).astype(int)

    print("accuracy: ", functional.accuracy(y_pred, y_true))
    print("precision: ", functional.precision(y_pred, y_true))
    print("recall: ", functional.recall(y_pred, y_true))
    print("f1: ", functional.f1_score(y_pred, y_true))
    