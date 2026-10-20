import numpy as np
import importlib

def main():
    p = np.array([0.1, 0.8, 0.4, 0.7, 0.9])

    functional = importlib.import_module("dl_lib.metrics.functional");

    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"threshold={threshold}: predictions=", functional.apply_threshold(p, threshold))