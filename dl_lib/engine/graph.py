import graphviz


def trace(value):
    nodes = [value]
    edges = []
    for i in value._prev:
        resultOfTrace = trace(i)

        nodes += resultOfTrace[0]
        edges.append((value, i))
        edges += resultOfTrace[1]

    return (nodes, edges)


def draw_dot(value):

    resultOfTrace = trace(value)

    dot = graphviz.Digraph(graph_attr={"rankdir": "LR"})

    for i in range(len(resultOfTrace[0])):
        dot.node(name=f"{i}", label=f"{resultOfTrace[0][i].data()}")

    # dot.node(name="a", label="a", shape="record")
    # dot.node(name="b", label="b", shape="record")
    # dot.node(name="a+b", label="+")
    # dot.edge("a", "a+b")
    # dot.edge("b", "a+b")

    svg_bytes = dot.pipe(format="svg")

    print(svg_bytes)
