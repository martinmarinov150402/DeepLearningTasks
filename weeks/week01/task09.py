import numpy as np

def main():
    scalar = 10

    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ]);

    print("Matrix + scalar 10: \n", np.add(scalar, matrix));

    rowVector = np.array([1, 2, 3]);
    print("Matrix + row vector: \n", np.add(rowVector, matrix));

    colVector = np.array([[1], [2]]);
    print("Matrix + column vector: \n", np.add(colVector, matrix));

    incompatibleVector = np.array([1, 2]);
    try:
        print("Matrix + incompatible vector: \n", np.add(incompatibleVector, matrix));
    except ValueError as e:
        print("Error adding incompatible vector to matrix:", e)