import logging
import math
import sys
import time

import pandas as pd
from numpy import random
from utils import plotting

from ppga import algorithms, base, log, tools


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
    if len(argv) < 4:
        print(f"USAGE: py {argv[0]} <T> <N> <G> <LOG-LEVEL>")
        exit(1)

    if len(argv) < 5:
        argv.append("success")

    logger = log.getUserLogger()
    logger.setLevel(argv[4].upper())

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
    toolbox.set_generation(tools.gen_permutation, range(len(towns)))
    toolbox.set_selection(tools.sel_tournament, tournsize=2)
    toolbox.set_crossover(tools.cx_one_point_ordered)
    toolbox.set_mutation(tools.mut_rotation)
    toolbox.set_evaluation(evaluate, towns)

    hall_of_fame = base.HallOfFame(5)

    # sequential execution
    start = time.perf_counter()
    best, stats = algorithms.elitist(
        toolbox=toolbox,
        population_size=N,
        keep=0.3,
        cxpb=0.7,
        mutpb=0.3,
        max_generations=G,
        hall_of_fame=hall_of_fame,
    )
    stime = time.perf_counter() - start
    logger.info(f"sequential best score: {best[0].fitness}")
    for i, ind in enumerate(hall_of_fame):
        logger.info(f"{i + 1}. {ind.values}")
    logger.info(f"sequential time: {stime}")

    # parallel execution
    hall_of_fame.clear()
    start = time.perf_counter()
    pbest, pstats = algorithms.pelitist(
        toolbox=toolbox,
        population_size=N,
        keep=0.3,
        cxpb=0.7,
        mutpb=0.3,
        max_generations=G,
        hall_of_fame=hall_of_fame,
    )
    ptime = time.perf_counter() - start
    logger.info(f"parallel best score: {pbest[0].fitness}")
    for i, ind in enumerate(hall_of_fame):
        logger.info(f"{i + 1}. {ind.values}")
    logger.info(f"parallel time: {ptime}")

    if stime / ptime >= 1.0:
        logger.log(15, f"speed up: {stime / ptime}")
    else:
        logger.warning(f"speed up: {stime / ptime}")

    # statistics data plot
    if logger.level <= logging.DEBUG:
        solution = max(best, key=lambda x: x.fitness)
        plotting.draw_graph(data, solution.chromosome)
        plotting.fitness_trend(stats)
        plotting.biodiversity_trend(stats)

        plotting.draw_graph(data, hall_of_fame[0].chromosome)
        plotting.fitness_trend(pstats)
        plotting.biodiversity_trend(pstats)

        plotting.evals(stats.evals)


if __name__ == "__main__":
    main(sys.argv)
