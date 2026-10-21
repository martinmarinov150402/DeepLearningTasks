import importlib

def main():

    functional = importlib.import_module("dl_lib.perceptrons.functional")

    print("Dataset:", functional.create_dataset(4))

    print("initialize_weights(0, 100):", functional.initialize_weights(1,100,1))
    print("initialize_weights(0, 10):", functional.initialize_weights(1,10,1))