import multiprocessing as mp
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from ppga import base


def main(argv: list[str]):
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    L = int(argv[1])
    chromosome = [i for i in range(L)]
    random.shuffle(chromosome)
    individual = base.Individual(chromosome, base.Fitness(weights=(1.0,)))
    population = [individual for _ in range(50000)]

    logger.info("----- SIZE ----")
    logger.info(f"chromosome size: {sys.getsizeof(chromosome)} bytes")
    logger.info(f"individual size: {sys.getsizeof(individual)} bytes")

    population_size = 0
    for i in population:
        population_size += sys.getsizeof(i)
    population_size += sys.getsizeof(population)
    logger.info(f"population size: {population_size / 1024.0 / 1024.0:.2f} MB")

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

    logger.info("----- SINGLE INDIVIDUAL ----")
    logger.info(f"mean time: {(np.mean(total)) * 1000.0:.5f} ms")
    logger.info(f"put mean time: {(np.mean(put_time)) * 1000.0:.5f} ms")
    logger.info(f"get mean time: {(np.mean(get_time)) * 1000.0:.5f} ms")

    logger.info(f"total put time: {(np.sum(put_time)) * 1000.0:.5f} ms")
    logger.info(f"total get time: {(np.sum(get_time)) * 1000.0:.5f} ms")
    logger.info(f"total time: {(np.sum(total)) * 1000.0:.5f} ms")

    # measuring all population at once
    start = time.perf_counter()
    buffer.put(population)
    put_end = time.perf_counter()
    population = buffer.get()
    end = time.perf_counter()

    logger.info("----- WHOLE POPULATION ----")
    logger.info(f"put population: {(put_end - start) * 1000.0:.5f} ms")
    logger.info(f"get population: {(end - put_end) * 1000.0:.5f} ms")
    logger.info(f"population: {(end - start) * 1000.0:.5f} ms")

    def chunk_test(chunksize: int):
        put_time = []
        get_time = []

        ops = 0
        for i in range(0, len(population), chunksize):
            ops += 1
            start = time.perf_counter()
            buffer.put(population[i : i + chunksize])
            put_end = time.perf_counter()
            chunk = buffer.get()
            end = time.perf_counter()

            assert len(chunk) <= chunksize

            put_time.append((put_end - start) * 1000.0)
            get_time.append((end - put_end) * 1000.0)

        logger.info(f"operations with chunksize {chunksize}: {ops}")

        return put_time, get_time, sys.getsizeof(chunk)

    # multiple chunks
    mean_put_times = []
    mean_get_times = []

    total_put_times = []
    total_get_times = []

    sizeof_chunks = []
    chunksizes = [50000, 25000, 12500, 6250, 3125, 1563, 782, 391, 196]
    for chunksize in chunksizes:
        put_time, get_time, sizeof_chunk = chunk_test(chunksize)

        mean_put_times.append(np.mean(put_time))
        mean_get_times.append(np.mean(get_time))

        total_put_times.append(np.sum(put_time))
        total_get_times.append(np.sum(get_time))

        sizeof_chunks.append(sizeof_chunk)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), layout="tight")
    axes[0].set_title("Chunk mean timings")
    axes[0].set_xlabel("Size of the chunk (bytes)")
    axes[0].set_ylabel("Operation time (ms)")
    axes[0].set_xscale("log")

    axes[0].plot(sizeof_chunks, mean_get_times, label="Mean get time")
    axes[0].plot(sizeof_chunks, mean_put_times, label="Mean put time")

    axes[0].grid()
    axes[0].legend()

    axes[1].set_title("Chunk total timings")
    axes[1].set_xlabel("Size of the chunk (bytes)")
    axes[1].set_ylabel("Operation time (ms)")
    axes[1].set_xscale("log")

    axes[1].plot(sizeof_chunks, total_get_times, label="Total get time")
    axes[1].plot(sizeof_chunks, total_put_times, label="Total put time")

    axes[1].grid()
    axes[1].legend()

    plt.show()
    # fig.savefig("../docs/images/queue_chunk.svg", format="svg", transparent=True)


if __name__ == "__main__":
    main(sys.argv)
