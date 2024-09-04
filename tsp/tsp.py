import sys


class Town:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def fitness(genome, distances):
    pass


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("USAGE: py tsp.py <T> <N> <G> <M>")
        exit(1)

    T = int(sys.argv[1])
    N = int(sys.argv[2])
    G = int(sys.argv[3])
    M = float(sys.argv[4])
