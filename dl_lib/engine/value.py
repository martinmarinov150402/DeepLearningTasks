class Value:
    __data = 0.0

    _op = ""

    def __init__(self, data):
        self.__data = data
        self._prev = []

    def __repr__(self) -> str:
        return f"Value(data={self.__data})"

    def __add__(self, other):
        newVal = Value(self.__data + other.__data)
        newVal._prev = [self, other]
        newVal._op = "+"
        return newVal

    def __mul__(self, other):
        newVal = Value(self.__data * other.__data)
        newVal._prev = [self, other]
        newVal._op = "*"
        return newVal

    def data(self):
        return self.__data
