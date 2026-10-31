from dl_lib.value.value import Value


def main():
    val1 = Value(1.0)
    val2 = Value(2.0)
    val3 = Value(3.0)

    print(val1)

    val4 = val1 + val3
    print(val4)

    val6 = val2 * val3
    print(val6)
