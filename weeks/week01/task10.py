import numpy as np

def main():
    A = np.array([
        [1, 2],
        [3, 4]
    ])

    B = np.array([
        [5, 6],
        [7, 8]
    ])

    elementwise_product = np.multiply(A, B);
    matrix_product = np.dot(A, B);

    print("Element-wise product: \n", elementwise_product);
    print("Matrix product: \n", matrix_product);

    print("The transpose of A: \n", np.transpose(A));