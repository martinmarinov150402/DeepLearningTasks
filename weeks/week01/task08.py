import numpy as np
import time

def main():

    size_of_array = 1000000
    rng = np.random.default_rng(seed=42)
    arr = np.array(rng.uniform(1, 10000, size_of_array));
    #arr = np.array(range(1000000))

    startWithFor = time.time()
    sum = 0
    for i in range(0, size_of_array):
        sum += arr[i]
    
    elapsedWithFor = time.time() - startWithFor

    startWithNumpy = time.time()
    sumV = np.sum(arr);
    elapsedWithNumpy = time.time() - startWithNumpy

    print(f"Sum with for loop: {sum}, elapsed time: {elapsedWithFor:.6f} seconds")
    print(f"Sum with numpy: {sumV}, elapsed time: {elapsedWithNumpy:.6f} seconds")
        
