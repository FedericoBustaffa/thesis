import math
import random
import sys
from functools import partial

import pandas as pd

from genetic import Genome
from genetic import GeneticAlgorithm
import plotting


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


def tournament(population: list[Genome]) -> list[int]:
    selected = []
    indices = [i for i in range(len(population))]

    for _ in range(len(population) // 2):
        first, second = random.choices(indices, k=2)
        while first == second:
            first, second = random.choices(indices, k=2)

        if population[first].fitness > population[second].fitness:
            selected.append(first)
            indices.remove(first)
        else:
            selected.append(second)
            indices.remove(second)

    return selected


def one_point_no_rep(father: list[int], mother: list[int]) -> tuple:
    crossover_point = random.randint(1, len(father) - 2)

    offspring1 = father[:crossover_point]
    offspring2 = father[crossover_point:]

    for gene in mother:
        if gene not in offspring1:
            offspring1.append(gene)
        else:
            offspring2.append(gene)

    return offspring1, offspring2


def rotation(offspring: list[int]) -> list[int]:
    a = random.randint(0, len(offspring))
    b = random.randint(0, len(offspring))

    while a == b:
        b = random.randint(0, len(offspring))

    first = a if a < b else b
    second = a if a > b else b
    offspring[first:second] = list(reversed(offspring[first:second]))

    return offspring


def merge_replace(population: list[Genome], offsprings: list[Genome]) -> list[Genome]:
    next_generation = []
    index = 0
    index1 = 0
    index2 = 0

    while (
        index < len(population)
        and index1 < len(population)
        and index2 < len(offsprings)
    ):
        if population[index1].fitness > offsprings[index2].fitness:
            next_generation.append(population[index1])
            index1 += 1
        else:
            next_generation.append(offsprings[index2])
            index2 += 1

        index += 1

    if index1 >= len(population):
        return next_generation
    elif index2 >= len(offsprings):
        next_generation.extend(population[index1 : len(population) - index2])

    return next_generation


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"USAGE: py {sys.argv[0]} <T> <N> <G> <M>")
        exit(1)

    data = pd.read_csv(f"../datasets/{sys.argv[1]}")
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
    ga.run(G)

    best = ga.get_best()
    print(f"best score: {best.fitness:.3f}")

    # drawing the graph
    plotting.draw_graph(data, best.chromosome)

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
