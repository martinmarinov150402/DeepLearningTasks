import numpy as np
import torch


def main():
    array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    tensor = torch.from_numpy(array)

    print("Numpy Array: ", array)
    print(f"dtype: {array.dtype}, shape: {array.shape}")
    print("Torch Tensor: ", tensor)
    print(f"dtype: {tensor.dtype}, shape: {tensor.shape}, device: {tensor.device}")

    newDevice = "cpu"
    if torch.cuda.is_available():
        newDevice = torch.device("cuda")
    elif torch.backends.mps.is_available():
        newDevice = torch.device("mps")
    else:
        newDevice = torch.device("cpu")

    print(f"Selected device: {newDevice}")
    tensor = tensor.to(newDevice)
    print(f"Tensor moved to device: {tensor.device}, dtype: {tensor.dtype}")
