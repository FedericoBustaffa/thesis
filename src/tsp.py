import math
import random
import sys
import time

import pandas as pd
from loguru import logger

from ppga import base
from ppga.algorithms import parallel, sequential
from ppga.tools import crossover, generation, mutation, replacement, selection
from utils import plotting


class Town:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def distance(t1: Town, t2: Town) -> float:
    return math.sqrt(math.pow(t1.x - t2.x, 2) + math.pow(t1.y - t2.y, 2))


def evaluate(chromosome, towns: list[Town]) -> tuple[float]:
    total_distance = 0.0
    for i in range(len(chromosome) - 1):
        total_distance += distance(towns[chromosome[i]], towns[chromosome[i + 1]])

    return (total_distance,)


def main(argv: list[str]):
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | {file}:{line} | <level>{level} - {message}</level>",
        level="TRACE",
        enqueue=True,
    )

    if len(argv) != 4:
        logger.error(f"USAGE: py {argv[0]} <T> <N> <G>")
        exit(1)

    data = pd.read_csv(f"datasets/towns_{argv[1]}.csv")
    x_coords = data["x"]
    y_coords = data["y"]
    towns = [Town(x, y) for x, y in zip(x_coords, y_coords)]

    # Initial population size
    N = int(argv[2])

    # Max generations
    G = int(argv[3])

    toolbox = base.ToolBox()
    toolbox.set_weights((-1.0,))
    toolbox.set_attributes(random.sample, range(len(towns)), len(towns))
    toolbox.set_generation(generation.iterate)
    toolbox.set_selection(selection.tournament, tournsize=2)
    toolbox.set_crossover(crossover.one_point_ordered, cxpb=0.7)
    toolbox.set_mutation(mutation.rotation, mutpb=0.3)
    toolbox.set_evaluation(evaluate, towns)
    toolbox.set_replacement(replacement.merge)

    hall_of_fame = base.HallOfFame(5)

    start = time.perf_counter()
    seq_best, seq_stats = sequential.generational(
        toolbox, N, G, hall_of_fame=hall_of_fame
    )
    sequential_time = time.perf_counter() - start
    logger.success(f"sequential best score: {seq_best[0].fitness}")
    logger.success(f"sequential total time: {sequential_time:.5f} seconds")
    print(hall_of_fame)

    hall_of_fame.clear()
    start = time.perf_counter()
    parallel_pop, parallel_stats = parallel.generational(
        toolbox, N, G, hall_of_fame=hall_of_fame
    )
    parallel_time = time.perf_counter() - start
    logger.success(f"parallel best score: {parallel_pop[0].fitness}")
    logger.success(f"parallel time: {parallel_time:.5f} seconds")
    print(hall_of_fame)

    # statistics data
    plotting.draw_graph(data, seq_best[0].chromosome)
    plotting.fitness_trend(seq_stats.best, seq_stats.mean, seq_stats.worst)

    plotting.draw_graph(data, parallel_pop[0].chromosome)
    plotting.fitness_trend(
        parallel_stats.best, parallel_stats.mean, parallel_stats.worst
    )


if __name__ == "__main__":
    main(sys.argv)
