import importlib
import numpy as np

def main():

    activations = importlib.import_module("dl_lib.activations.functional")

    scores = np.array([1., 2., 3.])

    print(f"scores: {scores}")

    softmax_my = activations.softmax(scores)
    print(f"softmax: {softmax_my}")


    print(f"sum of probabilities: {np.sum(softmax_my):.5f}")

    large_scores = [1000., 1001., 1002.]
    print(f"Large scores: {large_scores}")
    print(f"Naive softmax (unstable): {activations.naive_softmax(large_scores)}")
    print(f"Stable softmax: {activations.softmax(large_scores)}")




    