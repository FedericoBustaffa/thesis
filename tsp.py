import random
import time
import sys

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


def fitness(chromosome: np.ndarray, distances: np.ndarray) -> np.float64:
    total_distance = np.float64(0.0)
    for i in range(len(chromosome) - 1):
        total_distance += distances[chromosome[i]][chromosome[i + 1]]

    return np.float64(1.0 / total_distance)


def tournament(population, fitness_values):
    selected = [0 for _ in range(len(population) // 2)]
    indices = [i for i in range(len(population))]

    for i in range(len(selected)):
        first, second = random.choices(indices, k=2)
        # while first == second:
        #     first, second = random.choices(indices, k=2)

        if fitness_values[first] > fitness_values[second]:
            selected[i] = first
            indices.remove(first)
        else:
            selected[i] = second
            indices.remove(second)

    return selected


def one_point_no_rep(father, mother) -> tuple:
    crossover_point = random.randint(1, len(father) - 2)
    offspring1 = father[:crossover_point]
    offspring2 = father[crossover_point:]

    for gene in mother:
        if gene not in offspring1:
            offspring1.append(gene)
        else:
            offspring2.append(gene)

    return offspring1, offspring2


def rotation(offspring):
    indices = [i for i in range(len(offspring))]
    a, b = random.choices(indices, k=2)
    # while a == b:
    #     print("mutation conflict")
    #     b = random.choice(indices)
    first = a if a < b else b
    second = a if a > b else b

    head = offspring[:first]
    middle = reversed(offspring[first:second])
    tail = offspring[second:]
    head.extend(middle)
    head.extend(tail)
    offspring = head

    return offspring


def merge_replace(population, offsprings):
    next_generation = []
    index1 = 0
    index2 = 0

    while index1 < len(population) and index2 < len(offsprings):
        if population[index1].fitness > offsprings[index2].fitness:
            next_generation.append(population[index1])
            index1 += 1
        else:
            next_generation.append(offsprings[index2])
            index2 += 1

    if index1 >= len(population):
        next_generation.extend(offsprings[index2:])
    else:
        next_generation.extend(population[index1:])

    return next_generation[: len(population)]


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

    # useful data
    average_fitness = []
    best_fitness = []
    biodiversities = []
    timings = {
        "generation": 0.0,
        "evaluation": 0.0,
        "selection": 0.0,
        "crossover": 0.0,
        "mutation": 0.0,
        "replacement": 0.0,
    }

    ga = GeneticAlgorithm()

    # generate initial population
    start = time.perf_counter()
    ga.generation(N, generate, len(distances))
    end = time.perf_counter()
    timings["generation"] += end - start

    chromo, best = ga.get_best()
    print(f"first best: {best}")

    for g in range(G):
        # selection
        start = time.perf_counter()
        selected = ga.selection(tournament)
        end = time.perf_counter()
        timings["selection"] += end - start

        # crossover
        start = time.perf_counter()
        ga.crossover(one_point_no_rep)
        end = time.perf_counter()
        timings["crossover"] += end - start

        # mutation
        start = time.perf_counter()
        ga.mutation(rotation, mutation_rate)
        end = time.perf_counter()
        timings["mutation"] += end - start

        # offsprings evaluation
        start = time.perf_counter()
        ga.evaluation(fitness, distances)
        # ga.evaluation(fitness2, towns)
        end = time.perf_counter()
        timings["evaluation"] += end - start

        # replacement
        start = time.perf_counter()
        ga.replace(merge_replace)
        end = time.perf_counter()
        timings["replacement"] += end - start

        current_chromo, current_best = ga.get_best()
        if best.fitness < current_best:
            best = current_best
            chromo = current_chromo

        best_fitness.append(best)
        biodiversities.append(ga.biodiversity())
        average_fitness.append(ga.average_fitness())
        # if best.fitness == average_fitness[-1]:
        #     print(f"stopped at generation: {g}")
        #     break

    print(f"best solution: {best}")

    # drawing the graph
    plotting.draw_graph(data, chromo)

    # plotting data
    plotting.fitness_trend(average_fitness, best_fitness)
    plotting.biodiversity_trend(biodiversities)

    # timing
    plotting.timing(timings)

    for k in timings.keys():
        print(f"{k}: {timings[k]:.3f} seconds")
    print(f"total time: {sum(timings.values()):.3f} seconds")
