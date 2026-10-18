import importlib;
import numpy as np;

def main():
    preprocessing = importlib.import_module('dl_lib.preprocessing.functional');
    min_max_normalize = preprocessing.min_max_normalize;
    standardize = preprocessing.standardize;


    array = np.array([2., 4., 4., 4., 5., 5., 7., 9.]);

    normalized_array = min_max_normalize(array, 0, 1);
    standardized_array = standardize(array);

    print("Min-Max: ", normalized_array);
    print("Standardized: ", standardized_array);