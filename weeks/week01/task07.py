import numpy as np

def biggerThanSix(element):
    return element > 6
def main():
    matrix = np.array(
        [[ 1, 2, 3, 4],
         [ 5, 6, 7, 8],
         [ 9, 10, 11, 12]]
    )
    print("Second row: ", matrix[1]);
    print("Third column: ", matrix[:, 2]);
    print("Submatrix: \n", matrix[0:2, 1:3]);

    biggerThanSixVectorized = np.vectorize(biggerThanSix)

    boolMask = biggerThanSixVectorized(matrix)
    print("Boolean mask for more than 6: \n", boolMask);
    print("Values satisfying the mask: ", matrix[biggerThanSixVectorized(matrix)]);