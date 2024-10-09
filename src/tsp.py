import math
import random
import sys
import time

import numpy as np
import pandas as pd
from loguru import logger

from ppga import base, solver
from utils import plotting


class Town:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def generate(length: int):
    chromosome = [i for i in range(length)]
    random.shuffle(chromosome)

    return chromosome


def distance(t1: Town, t2: Town) -> float:
    return math.sqrt(math.pow(t1.x - t2.x, 2) + math.pow(t1.y - t2.y, 2))


def evaluate(chromosome, towns: list[Town]) -> tuple:
    total_distance = 0.0
    for i in range(len(chromosome) - 1):
        total_distance += distance(towns[chromosome[i]], towns[chromosome[i + 1]])

    time.sleep(0.0005)

    return (total_distance,)


def tournament(population: list[base.Individual]):
    selected = []
    indices = [i for i in range(len(population))]

    for _ in range(len(population) // 2):
        first, second = random.choices(indices, k=2)
        while first == second:
            first, second = random.choices(indices, k=2)

        if population[first] > population[second]:
            selected.append(population[first])
            indices.remove(first)
        else:
            selected.append(population[second])
            indices.remove(second)

    return selected


def couples_mating(population: list[base.Individual]) -> list[tuple]:
    indices = [i for i in range(len(population))]
    couples = []
    for _ in range(len(population) // 2):
        father, mother = random.sample(indices, k=2)
        couples.append((population[father], population[mother]))
        indices.remove(father)
        indices.remove(mother)

    return couples


def cx_one_point_ordered(father: list[int], mother: list[int]) -> tuple:
    crossover_point = random.randint(1, len(father) - 2)

    offspring1 = father[:crossover_point]
    offspring2 = father[crossover_point:]

    for gene in mother:
        if gene not in offspring1:
            offspring1.append(gene)
        else:
            offspring2.append(gene)

    return offspring1, offspring2


def mut_rotation(chromosome: list[int]):
    a = random.randint(0, len(chromosome) - 1)
    b = random.randint(0, len(chromosome) - 1)

    while a == b:
        b = random.randint(0, len(chromosome) - 1)

    first = a if a < b else b
    second = a if a > b else b
    chromosome[first:second] = reversed(chromosome[first:second])

    return chromosome


def merge(
    population: list[base.Individual], offsprings: list[base.Individual]
) -> list[base.Individual]:
    population = sorted(population, reverse=True)
    offsprings = sorted(offsprings, reverse=True)

    next_generation = []
    index = 0
    index1 = 0
    index2 = 0

    while (
        index < len(population)
        and index1 < len(population)
        and index2 < len(offsprings)
    ):
        if population[index1] > offsprings[index2]:
            next_generation.append(population[index1])
            index1 += 1
        else:
            next_generation.append(offsprings[index2])
            index2 += 1
        index += 1

    if index1 >= len(population):
        return next_generation
    elif index2 >= len(offsprings):
        next_generation[index:] = population[index1 : len(population) - index2]

    return next_generation


if __name__ == "__main__":
    if len(sys.argv) < 7:
        logger.error(f"USAGE: py {sys.argv[0]} <T> <N> <G> <C> <M> <W> <log_level>")
        exit(1)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | {file}:{line} | <level>{level} - {message}</level>",
        level=sys.argv[7].upper(),
        enqueue=True,
    )

    data = pd.read_csv(f"datasets/towns_{sys.argv[1]}.csv")
    towns = [Town(data["x"].iloc[i], data["y"].iloc[i]) for i in range(len(data))]

    # Initial population size
    N = int(sys.argv[2])

    # Max generations
    G = int(sys.argv[3])

    # crossover rate
    CR = float(sys.argv[4])

    # mutation rate
    MR = float(sys.argv[5])

    # number of workers
    W = int(sys.argv[6])

    toolbox = base.ToolBox()
    toolbox.set_fitness_weights(weights=(-1.0,))
    toolbox.set_generation(generate, len(towns))
    toolbox.set_selection(tournament)
    toolbox.set_mating(couples_mating)
    toolbox.set_crossover(cx_one_point_ordered, CR)
    toolbox.set_mutation(mut_rotation, MR)
    toolbox.set_evaluation(evaluate, towns)
    toolbox.set_replacement(merge)

    genetic_solver = solver.GeneticSolver()
    start = time.perf_counter()
    seq_best, seq_stats = genetic_solver.run(toolbox, N, G, base.Statistics())
    sequential_time = time.perf_counter() - start
    mean_eval_time = np.mean(toolbox.timings)

    queued_solver = solver.QueuedGeneticSolver(W)
    start = time.perf_counter()
    queue_best, queue_stats = queued_solver.run(toolbox, N, G, base.Statistics())
    queue_time = time.perf_counter() - start

    logger.trace(f"evaluation mean time: {mean_eval_time} seconds")
    logger.trace(f"evaluation total time: {np.sum(toolbox.timings)} seconds")
    logger.trace(f"evaluation max time: {max(toolbox.timings)}")
    logger.trace(f"evaluation max time: {min(toolbox.timings)}")

    # logger.success(f"sequential best score: {seq_best[0].fitness}")
    seq_t = sum(
        [
            seq_stats.timings["evaluation"],
            seq_stats.timings["crossover"],
            seq_stats.timings["mutation"],
        ]
    )
    queue_t = queue_stats.timings["parallel"]

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
            queue_stats.timings["crossover"],
            queue_stats.timings["mutation"],
            queue_stats.timings["evaluation"],
        ]
    )
    queue_sync_time = queue_stats.timings["parallel"] - pure_work_time

    logger.info(f"queue pure work time: {pure_work_time} seconds")
    logger.info(f"queue sync time: {queue_sync_time} seconds")
    logger.info(f"queue parallel time: {queue_stats.timings["parallel"]} seconds")

    # statistics data
    plotting.draw_graph(data, seq_best[0].chromosome)
    plotting.fitness_trend(seq_stats.best, seq_stats.worst)

    plotting.draw_graph(data, queue_best[0].chromosome)
    plotting.fitness_trend(queue_stats.best, queue_stats.worst)
