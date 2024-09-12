import random
import time
import sys

from functools import partial

import numpy as np
import pandas as pd

from genetic import GeneticAlgorithm
import plotting


def generate(length: int) -> np.ndarray:
    chromosome = np.array([i for i in range(length)])
    np.random.shuffle(chromosome)

    return chromosome


def distance(t1, t2) -> np.float64:
    return np.sqrt(np.pow(t1[0] - t2[0], 2) + np.pow(t1[1] - t2[1], 2))


def compute_distances(data: pd.DataFrame) -> np.ndarray:
    return np.array([[distance(t1, t2) for t2 in data.values] for t1 in data.values])


def fitness(distances: np.ndarray, chromosome: np.ndarray) -> np.float64:
    total_distance = np.float64(0.0)
    for i in range(len(chromosome) - 1):
        total_distance += distances[chromosome[i]][chromosome[i + 1]]

    return np.float64(1.0 / total_distance)


def tournament(population: np.ndarray, scores: np.ndarray) -> np.ndarray:
    selected = np.zeros(len(population) // 2, dtype=np.int32)
    indices = [i for i in range(len(population))]

    for i in range(len(selected)):
        first, second = random.choices(indices, k=2)
        # while first == second:
        #     first, second = random.choices(indices, k=2)

        if scores[first] > scores[second]:
            selected[i] = first
            indices.remove(first)
        else:
            selected[i] = second
            indices.remove(second)

    return selected


def one_point_no_rep(father: np.ndarray, mother: np.ndarray) -> tuple:
    crossover_point = random.randint(1, len(father) - 2)
    offspring1 = np.zeros(len(father))
    offspring2 = np.zeros(len(father))

    offspring1[:crossover_point] = father[:crossover_point]
    offspring2[crossover_point:] = father[crossover_point:]

    idx1 = crossover_point
    idx2 = 0
    for gene in mother:
        if gene not in offspring1[:crossover_point]:
            offspring1[idx1] = gene
            idx1 += 1
        else:
            offspring2[idx2] = gene
            idx2 += 1

    return offspring1.astype("int32"), offspring2.astype("int32")


def rotation(offspring: np.ndarray) -> np.ndarray:
    a = np.random.randint(0, len(offspring))
    b = np.random.randint(0, len(offspring))
    while a == b:
        b = np.random.randint(0, len(offspring))

    first = a if a < b else b
    second = a if a > b else b
    offspring[first:second] = np.array(list(reversed(offspring[first:second])))

    return offspring


def merge_replace(
    population: list[np.ndarray],
    scores: np.ndarray,
    offsprings: list[np.ndarray],
    offsprings_scores: np.ndarray,
):
    next_generation = []
    next_generation_scores = np.zeros(len(population))
    index = 0
    index1 = 0
    index2 = 0

    while index < len(population) and index2 < len(offsprings):
        if scores[index1] > offsprings_scores[index2]:
            next_generation.append(population[index1])
            next_generation_scores[index] = scores[index1]
            index1 += 1
        else:
            next_generation.append(offsprings[index2])
            next_generation_scores[index] = offsprings_scores[index2]
            index2 += 1

        index += 1
        # print(f"{index}: {next_generation_scores[index]}")

    if index1 >= len(population):
        return next_generation, next_generation_scores
    elif index2 >= len(offsprings):
        next_generation.extend(population[index:])
        next_generation_scores[index:] = scores[index:]

    return next_generation, next_generation_scores


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"USAGE: py {sys.argv[0]} <T> <N> <G> <M>")
        exit(1)

    data = pd.read_csv(sys.argv[1])
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
        generate_func,
        fitness_func,
        tournament,
        one_point_no_rep,
        rotation,
        mutation_rate,
        merge_replace,
    )

    start = time.perf_counter()
    ga.run(G)
    end = time.perf_counter()
    print(f"time: {end - start:.3f} seconds")

    chromosome, score = ga.get_best()
    print(f"best score: {score:.3f}")

    # drawing the graph
    plotting.draw_graph(data, chromosome)

    # statistics data
    average_fitness = ga.get_average_fitness()
    best_fitness = ga.get_best_fitness()
    biodiversity = ga.get_biodiversity()

    plotting.fitness_trend(average_fitness, best_fitness)
    plotting.biodiversity_trend(biodiversity)

    # timing
    timings = ga.get_timings()
    plotting.timing(timings)

    for k in timings.keys():
        print(f"{k}: {timings[k]:.3f} seconds")
    print(f"total time: {sum(timings.values()):.3f} seconds")
