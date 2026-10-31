def and_dataset(): 
    return [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]

def or_dataset():
    return [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]

def xor_dataset():
    return [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

def nand_dataset():
    return [([0, 0], 1), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

def create_dataset(n):
    return [(x, x * 2) for x in range(n)]


