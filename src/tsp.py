import math
import random
import sys
import time

import pandas as pd
from loguru import logger

from ppga import algorithms, base, tools
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
    toolbox.set_generation(tools.iterate)
    toolbox.set_selection(tools.tournament, tournsize=2)
    toolbox.set_crossover(tools.one_point_ordered)
    toolbox.set_mutation(tools.rotation)
    toolbox.set_evaluation(evaluate, towns)
    toolbox.set_replacement(tools.merge)

    hall_of_fame = base.HallOfFame(5)

    start = time.perf_counter()
    seq_best, seq_stats = algorithms.sga(
        toolbox, N, 0.7, 0.3, G, hall_of_fame=hall_of_fame
    )
    sequential_time = time.perf_counter() - start
    logger.success(f"sequential best score: {seq_best[0].fitness}")
    logger.success(f"sequential total time: {sequential_time:.5f} seconds")
    for i, ind in enumerate(hall_of_fame):
        logger.info(f"{i}. {ind.values}")

    hall_of_fame.clear()
    start = time.perf_counter()
    parallel_pop, parallel_stats = algorithms.psga(
        toolbox, N, 0.7, 0.3, G, hall_of_fame=hall_of_fame
    )
    parallel_time = time.perf_counter() - start
    logger.success(f"parallel best score: {parallel_pop[0].fitness}")
    logger.success(f"parallel time: {parallel_time:.5f} seconds")
    for i, ind in enumerate(hall_of_fame):
        logger.info(f"{i}. {ind.values}")

    speed_up = sequential_time / parallel_time
    if speed_up <= 1.0:
        logger.warning(f"speed up: {speed_up:.3f}")
    else:
        logger.success(f"speed up: {speed_up:.3f}")

    # statistics data
    plotting.draw_graph(data, seq_best[0].chromosome)
    plotting.fitness_trend(seq_stats)
    plotting.biodiversity_trend(seq_stats)

    plotting.draw_graph(data, parallel_pop[0].chromosome)
    plotting.fitness_trend(parallel_stats)
    plotting.biodiversity_trend(parallel_stats)


if __name__ == "__main__":
    main(sys.argv)
