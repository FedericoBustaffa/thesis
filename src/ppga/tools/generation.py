def repeat(generator, n: int, length: int):
    return [[generator() for _ in range(length)] for _ in range(n)]


def iterate(generator, n: int):
    return [generator() for _ in range(n)]
