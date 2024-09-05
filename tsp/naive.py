import random
import sys

import pandas as pd

from ppga.genetic import pure


def chromosome_generation(values: list) -> list:
    chromosome = values
    random.shuffle(chromosome)

    return chromosome


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"USAGE: py {sys.argv[0]} <T> <N> <G> <M>")
        exit(1)

    try:
        towns = pd.read_csv(sys.argv[1])
    except FileNotFoundError:
        print(f"File {sys.argv[1]} not found")
        exit(1)

    # Initial population size
    N = int(sys.argv[2])

    # Max generations
    G = int(sys.argv[3])

    # Mutation rate
    M = float(sys.argv[4])

    chromosome_values = [i for i in range(len(towns))]
    population = generate(N, chromosome_generation, chromosome_values)
    print(population)
