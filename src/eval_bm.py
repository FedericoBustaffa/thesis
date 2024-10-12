import random
import sys
import time

import numpy as np
import pandas as pd
from loguru import logger

import tsp


def main(argv: list[str]):
    data = pd.read_csv(f"datasets/towns_{argv[1]}.csv")
    x_coords = data["x"]
    y_coords = data["y"]
    towns = [tsp.Town(x, y) for x, y in zip(x_coords, y_coords)]

    # chromosome lenght
    L = int(argv[1])
    N = 10000
    chromosome = [i for i in range(L)]

    population = []
    for i in range(N):
        random.shuffle(chromosome)
        population.append(chromosome)

    # evaluation
    timings = []
    for chromosome in population:
        start = time.perf_counter()
        tsp.evaluate(chromosome, towns)
        end = time.perf_counter()

        timings.append((end - start) * 1000.0)

    logger.info(f"mean evaluation time: {np.mean(timings):.5f} ms")
    logger.info(f"total evaluation time: {np.sum(timings):.5f} ms")


if __name__ == "__main__":
    main(sys.argv)
