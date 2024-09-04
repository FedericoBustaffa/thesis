import sys

import pandas as pd


def fitness(genome, distances):
    pass


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("USAGE: py tsp.py <T> <N> <G> <M>")
        exit(1)

    try:
        towns = pd.read_csv(sys.argv[1])
    except FileNotFoundError:
        print(f"File {sys.argv[1]} not found")
        exit(1)

    print(towns)

    N = int(sys.argv[2])
    G = int(sys.argv[3])
    M = float(sys.argv[4])
