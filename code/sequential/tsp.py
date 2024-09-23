import sys
from functools import partial

import numpy as np
import pandas as pd
from genetic import GeneticAlgorithm

from utils import plotting


def generate(length: int) -> np.ndarray:
    chromosome = np.array([i for i in range(length)])
    np.random.shuffle(chromosome)

    return chromosome


def distance(t1, t2) -> float:
    return np.sqrt(np.pow(t1[0] - t2[0], 2) + np.pow(t1[1] - t2[1], 2))


def compute_distances(data: pd.DataFrame) -> np.ndarray:
    return np.array([[distance(t1, t2) for t2 in data.values] for t1 in data.values])


def fitness(distances: np.ndarray, chromosome: np.ndarray):
    total_distance = 0.0
    for i in range(len(chromosome) - 1):
        total_distance += distances[chromosome[i]][chromosome[i + 1]]

    return 1.0 / total_distance


def tournament(scores: np.ndarray) -> list[int]:
    selected = []
    indices = [i for i in range(len(scores))]

    for _ in range(len(scores) // 2):
        first, second = np.random.choice(indices, size=2)
        while first == second:
            first, second = np.random.choice(indices, size=2)

        if scores[first] > scores[second]:
            selected.append(first)
            indices.remove(first)
        else:
            selected.append(second)
            indices.remove(second)

    return selected


def one_point_no_rep(father: np.ndarray, mother: np.ndarray) -> tuple:
    crossover_point = np.random.randint(1, len(father) - 1)

    offspring1 = father[:crossover_point]
    offspring2 = father[crossover_point:]

    tail1 = np.zeros(father[crossover_point:].size, np.int64)
    tail2 = np.zeros(father[:crossover_point].size, np.int64)
    idx1 = 0
    idx2 = 0
    for gene in mother:
        if gene not in offspring1:
            tail1[idx1] = gene
            idx1 += 1
        else:
            tail2[idx2] = gene
            idx2 += 1

    return np.append(offspring1, tail1), np.append(offspring2, tail2)


def rotation(offspring: np.ndarray) -> np.ndarray:
    a = np.random.randint(0, len(offspring))
    b = np.random.randint(0, len(offspring))

    while a == b:
        b = np.random.randint(0, len(offspring))

    first = a if a < b else b
    second = a if a > b else b
    offspring[first:second] = np.flip(offspring[first:second])[:]

    return offspring


def merge_replace(
    population: np.ndarray,
    scores1: np.ndarray,
    offsprings: np.ndarray,
    scores2: np.ndarray,
) -> tuple:

    sort_indices = np.flip(np.argsort(scores1))
    population = np.array([population[i] for i in sort_indices])
    scores1 = scores1[sort_indices]

    sort_indices = np.flip(np.argsort(scores2))
    offsprings = np.array([offsprings[i] for i in sort_indices])
    scores2 = scores2[sort_indices]

    next_generation = np.zeros(population.shape, dtype=np.int64)
    next_gen_scores = np.zeros(scores1.shape, dtype=np.float64)
    index = 0
    index1 = 0
    index2 = 0

    while (
        index < len(population)
        and index1 < len(population)
        and index2 < len(offsprings)
    ):
        if scores1[index1] > scores2[index2]:
            next_generation[index] = population[index1]
            next_gen_scores[index] = scores1[index1]
            index1 += 1
        else:
            next_generation[index] = offsprings[index2]
            next_gen_scores[index] = scores2[index2]
            index2 += 1

        index += 1

    if index1 >= len(population):
        return next_generation, next_gen_scores
    elif index2 >= len(offsprings):
        next_generation[index:] = population[index1 : len(population) - index2]
        next_gen_scores[index:] = scores1[index1 : len(scores1) - index2]
        # next_generation.extend(population[index1 : len(population) - index2])
        # next_gen_scores.extend(scores1[index1 : len(scores1) - index2])

    return np.array(next_generation), np.array(next_gen_scores)


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

    ga = GeneticAlgorithm(
        N,
        len(data),
        generate_func,
        fitness_func,
        tournament,
        one_point_no_rep,
        rotation,
        mutation_rate,
        merge_replace,
    )
    ga.run(G)

    print(f"best score: {ga.best_score:.3f}")

    # drawing the graph
    plotting.draw_graph(data, ga.best)

    # statistics data
    plotting.fitness_trend(ga.average_fitness, ga.best_fitness)
    # plotting.biodiversity_trend(ga.biodiversity)

    # timing
    plotting.timing(ga.timings)

    for k in ga.timings.keys():
        print(f"{k}: {ga.timings[k]:.3f} seconds")
    print(f"total time: {sum(ga.timings.values()):.3f} seconds")
