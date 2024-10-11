import multiprocessing as mp
import random
import sys
import time

import numpy as np
from loguru import logger

from ppga import base, solver


def main(argv: list[str]):
    L = int(argv[1])
    chromosome = [i for i in range(L)]
    random.shuffle(chromosome)
    individual = base.Individual(chromosome, base.Fitness(weights=(1.0,)))
    population = [individual for _ in range(10000)]

    logger.debug(f"chromosome size: {sys.getsizeof(chromosome)} bytes")
    logger.debug(f"individual size: {sys.getsizeof(individual)} bytes")

    population_size = 0
    for i in population:
        population_size += sys.getsizeof(i)
    population_size += sys.getsizeof(population)
    logger.debug(f"population size: {population_size / 1024.0 / 1024.0:.2f} MB")

    buffer = mp.Queue()

    total = []
    put_time = []
    get_time = []
    for individual in population:
        start = time.perf_counter()
        buffer.put(individual)
        put_end = time.perf_counter()
        individual = buffer.get()
        end = time.perf_counter()

        total.append(end - start)
        put_time.append(put_end - start)
        get_time.append(end - put_end)

    logger.debug(f"mean time: {(np.mean(total)) * 1000.0:.5f} ms")
    logger.debug(f"put mean time: {(np.mean(put_time)) * 1000.0:.5f} ms")
    logger.debug(f"get mean time: {(np.mean(get_time)) * 1000.0:.5f} ms")

    logger.debug(f"total time: {(np.sum(total)) * 1000.0:.5f} ms")
    logger.debug(f"total put time: {(np.sum(put_time)) * 1000.0:.5f} ms")
    logger.debug(f"total get time: {(np.sum(get_time)) * 1000.0:.5f} ms")

    # measuring all population at once
    start = time.perf_counter()
    buffer.put(population)
    put_end = time.perf_counter()
    population = buffer.get()
    end = time.perf_counter()

    logger.debug(f"population: {(end - start) * 1000.0:.5f} ms")
    logger.debug(f"put population: {(put_end - start) * 1000.0:.5f} ms")
    logger.debug(f"get population: {(end - put_end) * 1000.0:.5f} ms")


if __name__ == "__main__":
    main(sys.argv)
