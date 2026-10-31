from dl_lib.engine.value import Value


def main():
    v1 = Value(2.0)
    v2 = Value(-3.0)

    v3 = Value(10.0)
    v4 = Value(-6.0)

    add = v3 + v4

    mul = v1 * v2

    print(add)
    print(add._op)
    print(add._prev)

    print(mul)
    print(mul._op)
    print(mul._prev)
