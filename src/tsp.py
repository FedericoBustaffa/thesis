import math
import random
import sys
import time

import pandas as pd
from loguru import logger

from ppga import base, solver
from ppga.tools import crossover, mutate, select
from utils import plotting


class Town:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def generate(length: int) -> list[int]:
    chromosome = [i for i in range(length)]
    random.shuffle(chromosome)

    return chromosome


def distance(t1: Town, t2: Town) -> float:
    return math.sqrt(math.pow(t1.x - t2.x, 2) + math.pow(t1.y - t2.y, 2))


def evaluate(chromosome, towns: list[Town]) -> tuple[float]:
    total_distance = 0.0
    for i in range(len(chromosome) - 1):
        total_distance += distance(towns[chromosome[i]], towns[chromosome[i + 1]])

    # wasting time
    for i in range(50000):
        random.random()

    return (total_distance,)


def main(argv: list[str]):
    if len(argv) < 5:
        logger.error(f"USAGE: py {argv[0]} <T> <N> <G> <W> <log_level>")
        exit(1)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | {file}:{line} | <level>{level} - {message}</level>",
        level=argv[5].upper(),
        enqueue=True,
    )

    data = pd.read_csv(f"datasets/towns_{argv[1]}.csv")
    x_coords = data["x"]
    y_coords = data["y"]
    towns = [Town(x, y) for x, y in zip(x_coords, y_coords)]

    # Initial population size
    N = int(argv[2])

    # Max generations
    G = int(argv[3])

    # crossover rate
    cxpb = 0.8

    # mutation rate
    mutpb = 0.2

    # number of workers
    W = int(argv[4])

    toolbox = base.ToolBox()
    toolbox.set_fitness((-1.0,))
    toolbox.set_generation(generate, len(towns))
    toolbox.set_selection(select.tournament, tournsize=3)
    toolbox.set_crossover(crossover.one_point_ordered, cxpb=cxpb)
    toolbox.set_mutation(mutate.rotation, mutpb=mutpb)
    toolbox.set_evaluation(evaluate, towns)

    genetic_solver = solver.GeneticSolver()
    start = time.perf_counter()
    seq_best, seq_stats = genetic_solver.run(toolbox, N, G)
    sequential_time = time.perf_counter() - start

    queued_solver = solver.QueuedGeneticSolver(W)
    start = time.perf_counter()
    queue_best, queue_stats = queued_solver.run(toolbox, N, G, base.Statistics())
    queue_time = time.perf_counter() - start

    # logger.success(f"sequential best score: {seq_best[0].fitness}")
    seq_t = sum(
        [
            seq_stats["evaluation"],
            seq_stats["crossover"],
            seq_stats["mutation"],
        ]
    )
    queue_t = queue_stats["parallel"]

    logger.info(f"sequential total time: {sequential_time:.5f} seconds")
    logger.info(f"to parallelize time: {seq_t:.5f} seconds")

    logger.info("-" * 50)
    # logger.success(f"queue best score: {queue_best[0].fitness}")
    logger.info(f"queue total time: {queue_time:.5f} seconds")
    if seq_t / queue_t > 1.0:
        logger.success(f"queue solver core speed up: {seq_t / queue_t:.5f}")
    else:
        logger.warning(f"queue solver core speed up: {seq_t / queue_t:.5f}")

    if sequential_time / queue_time > 1.0:
        logger.success(f"queue total speed up: {sequential_time / queue_time:.5f}")
    else:
        logger.warning(f"queue total speed up: {sequential_time / queue_time:.5f}")

    pure_work_time = sum(
        [
            queue_stats["crossover"],
            queue_stats["mutation"],
            queue_stats["evaluation"],
        ]
    )
    queue_sync_time = queue_stats["parallel"] - pure_work_time

    logger.info(f"queue pure work time: {pure_work_time} seconds")
    logger.info(f"queue sync time: {queue_sync_time} seconds")
    logger.info(f"queue parallel time: {queue_stats['parallel']} seconds")

    # statistics data
    plotting.draw_graph(data, seq_best[0].chromosome)
    plotting.fitness_trend(seq_stats.best, seq_stats.worst)

    plotting.draw_graph(data, queue_best[0].chromosome)
    plotting.fitness_trend(queue_stats.best, queue_stats.worst)

    plotting.timing({"sync": queue_sync_time, "pure": pure_work_time})


if __name__ == "__main__":
    main(sys.argv)
