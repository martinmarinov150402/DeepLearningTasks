from dl_lib.engine.graph import *
from dl_lib.engine.value import Value


def main():

    val1 = Value(-3.0)
    val2 = Value(2.0)

    val6 = val1 * val2
    val10 = Value(10)

    val4 = val6 + val10

    traceResult = trace(val4)

    print("Number of nodes: ", len(traceResult[0]))
    print("Number of edges: ", len(traceResult[1]))

    draw_dot(val4)
