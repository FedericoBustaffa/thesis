import math
import random
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
from numba import njit
from seq import GeneticAlgorithm
from sm_algorithm import SharedMemoryGeneticAlgorithm

from utils import plotting


def generate(length: int) -> list[int]:
    chromosome = [i for i in range(length)]
    random.shuffle(chromosome)

    return chromosome


def distance(t1, t2) -> float:
    return math.sqrt(math.pow(t1[0] - t2[0], 2) + math.pow(t1[1] - t2[1], 2))


def compute_distances(data: pd.DataFrame) -> list[list[float]]:
    return [[distance(t1, t2) for t2 in data.values] for t1 in data.values]


def fitness(distances: list[list[float]], chromosome: list[int]) -> float:
    total_distance = 0.0
    for i in range(len(chromosome) - 1):
        total_distance += distances[chromosome[i]][chromosome[i + 1]]

    return 1.0 / total_distance


def tournament(scores: list[float]) -> list[int]:
    selected = []
    indices = [i for i in range(len(scores))]

    for _ in range(len(scores) // 2):
        first, second = random.choices(indices, k=2)

        if scores[first] > scores[second]:
            selected.append(first)
            indices.remove(first)
        else:
            selected.append(second)
            indices.remove(second)

    return selected


@njit
def one_point_no_rep(father, mother) -> tuple:

    crossover_point = random.randint(1, len(father) - 2)

    offspring1 = list(father[:crossover_point])
    offspring2 = list(father[crossover_point:])

    for gene in mother:
        if gene not in offspring1:
            offspring1.append(gene)
        else:
            offspring2.append(gene)

    return offspring1, offspring2


def rotation(offspring: list[int]) -> list[int]:
    a = np.random.randint(0, len(offspring))
    b = np.random.randint(0, len(offspring))

    while a == b:
        b = np.random.randint(0, len(offspring))

    first = a if a < b else b
    second = a if a > b else b
    offspring[first:second] = list(reversed(offspring[first:second]))

    return offspring


def merge_replace(population, scores1, offsprings, scores2):

    population, scores1 = (
        list(t)
        for t in zip(
            *sorted(zip(population, scores1), key=lambda x: x[1], reverse=True)
        )
    )

    offsprings, scores2 = (
        list(t)
        for t in zip(
            *sorted(zip(offsprings, scores2), key=lambda x: x[1], reverse=True)
        )
    )

    next_generation = []
    next_gen_scores = []
    index = 0
    index1 = 0
    index2 = 0

    while (
        index < len(population)
        and index1 < len(population)
        and index2 < len(offsprings)
    ):
        if scores1[index1] > scores2[index2]:
            next_generation.append(population[index1])
            next_gen_scores.append(scores1[index1])
            index1 += 1
        else:
            next_generation.append(offsprings[index2])
            next_gen_scores.append(scores2[index2])
            index2 += 1

        index += 1

    if index1 >= len(population):
        return next_generation, next_gen_scores
    elif index2 >= len(offsprings):
        next_generation.extend(population[index1 : len(population) - index2])
        next_gen_scores.extend(scores1[index1 : len(scores1) - index2])

    return next_generation, next_gen_scores


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"USAGE: py {sys.argv[0]} <T> <N> <G> <M>")
        exit(1)

    data = pd.read_csv(f"datasets/towns_{sys.argv[1]}.csv")
    distances = compute_distances(data)

    # Initial population size
    N = int(sys.argv[2])

    # Max generations
    G = int(sys.argv[3])

    # Mutation rate
    mutation_rate = float(sys.argv[4])

    generate_func = partial(generate, len(distances))
    fitness_func = partial(fitness, distances)

    pga = SharedMemoryGeneticAlgorithm(
        N,
        generate_func,
        fitness_func,
        tournament,
        one_point_no_rep,
        rotation,
        mutation_rate,
        merge_replace,
        workers_num=20,
    )

    start = time.perf_counter()
    pga.run(G)
    print(f"total time: {time.perf_counter() - start} seconds")

    print(f"best score: {pga.best_score:.3f}")

    # # drawing the graph
    plotting.draw_graph(data, pga.best)

    # # statistics data
    plotting.fitness_trend(pga.average_fitness, pga.best_fitness)
    plotting.biodiversity_trend(pga.biodiversity)

    # timing
    plotting.timing(pga.timings)

    for k in pga.timings.keys():
        print(f"{k}: {pga.timings[k]:.3f} seconds")
