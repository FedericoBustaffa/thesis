import random
import sys
import math

import matplotlib.pyplot as plt
import pandas as pd

import pure


class Town:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def distance(t1: Town, t2: Town) -> float:
    return math.sqrt(math.pow(t1.x - t2.x, 2) + math.pow(t1.y - t2.y, 2))


def fitness(chromosome: list[int], distances: list[list[float]]) -> float:
    total_distance = 0
    for i in range(len(distances) - 1):
        total_distance += distances[chromosome[i]][chromosome[i + 1]]

    return 1 / total_distance


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"USAGE: py {sys.argv[0]} <T> <N> <G> <M>")
        exit(1)

    data = pd.read_csv(sys.argv[1])
    towns = [Town(data["x"][i], data["y"][i]) for i in range(len(data))]
    distances = [[distance(t1, t2) for t2 in towns] for t1 in towns]

    # Initial population size
    N = int(sys.argv[2])

    # Max generations
    G = int(sys.argv[3])

    # Mutation rate
    M = float(sys.argv[4])

    # generate initial population
    population = pure.generate(N, len(towns))
    print(f"population size: {len(population)}")

    population = pure.evaluation(population, fitness, distances)
    best = population[0]
    print(f"first best: {best.fitness}")
    for i in population:
        print(i)

    for g in range(G):
        selected = pure.selection(population)
        print(f"selected: {len(selected)}")
        for s in selected:
            print(s)

        offsprings = pure.crossover(selected)
        print(f"generated offsprings: {len(offsprings)}")
        for child in offsprings:
            print(child)

        offsprings = pure.mutation(offsprings, M)
        print("mutated offsprings")
        for child in offsprings:
            print(child)

        offsprings = pure.evaluation(offsprings, fitness, distances)
        print("offsprings evaluation")
        for child in offsprings:
            print(child)

        population = pure.replace(population, offsprings)
        print(f"replace: {len(population)} individuals")
        for i in population:
            print(i)

        if best.fitness < population[0].fitness:
            best = population[0]

    print(f"Best solution: {best.fitness}")

    # drawing the graph
    x = [towns[i].x for i in best.chromosome]
    y = [towns[i].y for i in best.chromosome]

    plt.figure(figsize=(12, 6))
    plt.title("Best path found")
    plt.xlabel("X coordinates")
    plt.ylabel("Y coordinates")

    plt.scatter(x, y, label="Towns")
    plt.plot(x, y, c="k", label="Path")

    plt.legend()
    plt.show()

    # plotting data
    # plt.figure(figsize=(12, 6))
    # generations = [g for g in range(len(average_fitness))]
    # plt.title("Fitness through generations")
    # plt.xlabel("Generations")
    # plt.ylabel("Fitness")

    # plt.plot(generations, average_fitness, label="Average fitness")
    # plt.plot(generations, best_fitness, label="Best fitness")

    # plt.grid()
    # plt.legend()
    # plt.show()

    # timing
    # plt.figure(figsize=(12, 6))
    # plt.title("Timing")
    # plt.pie([v for v in timings.values()], labels=[k for k in timings.keys()])
    # plt.show()
