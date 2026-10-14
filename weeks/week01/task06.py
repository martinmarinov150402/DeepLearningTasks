import numpy as np
def main():
    arr = np.array([1, 2, 3, 4, 5]);
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6]]);
    triDim = np.array([[
            [ 0 , 1,  2,  3],
            [ 4 , 5,  6,  7],
            [ 8 , 9, 10, 11]
        ],
        [
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]
        ]
    ]);

    print(f"1D array: shape={arr.shape}, dtype={arr.dtype}, ndim={arr.ndim}, size={arr.size}")
    print(arr);
    print(f"2D array: shape={matrix.shape}, dtype={matrix.dtype}, ndim={matrix.ndim}, size={matrix.size}")
    print(matrix);
    print(f"3D array: shape={triDim.shape}, dtype={triDim.dtype}, ndim={triDim.ndim}, size={triDim.size}")
    print(triDim);
