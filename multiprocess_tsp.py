import math
import sys
import time

import pandas as pd

import genetic
import plotting
from tsp import Town
from evaluation import PipeEvaluator


def distance(t1: Town, t2: Town) -> float:
    return math.sqrt(math.pow(t1.x - t2.x, 2) + math.pow(t1.y - t2.y, 2))


def fitness(chromosome: list[int], distances: list[list[float]]) -> float:
    total_distance = 0
    for i in range(len(distances) - 1):
        total_distance += distances[chromosome[i]][chromosome[i + 1]]

    return 1 / total_distance


def fitness2(chromosome: list[int], towns: list[Town]) -> float:
    total_distance = 0
    for i in range(len(towns) - 1):
        total_distance += distance(towns[chromosome[i]], towns[chromosome[i + 1]])

    return 1 / total_distance


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"USAGE: py {sys.argv[0]} <T> <N> <G> <M>")
        exit(1)

    data = pd.read_csv(sys.argv[1])
    x = data["x"]
    y = data["y"]
    towns = [Town(x.iloc[i], y.iloc[i]) for i in range(len(data))]
    distances = [[distance(t1, t2) for t2 in towns] for t1 in towns]

    # Initial population size
    N = int(sys.argv[2])

    # Max generations
    G = int(sys.argv[3])

    # Mutation rate
    M = float(sys.argv[4])

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

    # generate initial population
    start = time.perf_counter()
    population = genetic.generate(N, len(towns))
    end = time.perf_counter()
    timings["generation"] += end - start

    evaluator = PipeEvaluator(fitness, distances)
    start = time.perf_counter()
    evaluator.evaluate(population)
    end = time.perf_counter()
    timings["evaluation"] += end - start

    best = population[0]
    print(f"first best: {best.fitness}")

    for g in range(G):
        biodiversities.append(genetic.biodiversity(population))
        average_fitness.append(sum([i.fitness for i in population]) / len(population))

        # selection
        start = time.perf_counter()
        selected = genetic.selection(population)
        end = time.perf_counter()
        timings["selection"] += end - start

        # crossover
        start = time.perf_counter()
        offsprings = genetic.crossover(selected)
        end = time.perf_counter()
        timings["crossover"] += end - start

        # mutation
        start = time.perf_counter()
        offsprings = genetic.mutation(offsprings, M)
        end = time.perf_counter()
        timings["mutation"] += end - start

        # offsprings evaluation
        start = time.perf_counter()
        evaluator.evaluate(offsprings)
        end = time.perf_counter()
        timings["evaluation"] += end - start

        # replacement
        start = time.perf_counter()
        population = genetic.replace(population, offsprings)
        end = time.perf_counter()
        timings["replacement"] += end - start

        if best.fitness < population[0].fitness:
            best = population[0]

        best_fitness.append(best.fitness)

    print(f"best solution: {best.fitness}")
    evaluator.shutdown()

    # drawing the graph
    plotting.draw_graph(towns, best)

    # plotting data
    plotting.fitness_trend(average_fitness, best_fitness)
    plotting.biodiversity_trend(biodiversities)

    # timing
    plotting.timing(timings)

    for k in timings.keys():
        print(f"{k}: {timings[k]:.3f} seconds")
    print(f"total time: {sum(timings.values()):.3f} seconds")
