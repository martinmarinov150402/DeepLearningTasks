import numpy as np
import importlib

def main():
    losses = importlib.import_module("dl_lib.losses.functional")

    y_true = np.array([1., 0., 1.])
    p = np.array([0.9, 0.2, 0.6])
    print(f"Binary cross-entropy on y_true={y_true}, p={p}: {losses.binary_cross_entropy(p, y_true):.4f}")

    y_true_c = np.array([1])
    p_c = np.array([0.999])
    p_n_c = np.array([0.001])

    print(f"y_true=1, p=0.999 (confidently right): loss={losses.binary_cross_entropy(p_c, y_true_c):.4f}")
    print(f"y_true = 1, p = 0.001 (confidently wrong): loss={losses.binary_cross_entropy(p_n_c, y_true_c):.4f}")

    p_mc = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6]
    ])

    t = np.array([0, 1, 2])


    print(f"Multiclass cross-entropy on y_true=[0 1 2]: {losses.categorical_cross_entropy(t, p_mc):.4f}")

    p_conf_right = np.array([
        [0.999, 0.0005, 0.0005]
    ])

    p_conf_wrong = np.array([
        [0.001, 0.001, 0.998]
    ])

    print(f"y_true=0, p_true=0.999 (confidently right): loss={losses.categorical_cross_entropy([0], p_conf_right):.4f}")
    print(f"y_true=0, p_true=0.001 (confidently wrong): loss={losses.categorical_cross_entropy([0], p_conf_wrong):.4f}")