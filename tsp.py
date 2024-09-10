import math
import random
import sys
import time

import pandas as pd

from evaluation import PipeEvaluator
from crossover import PipeCrossover
import genetic
import plotting


class Town:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def generate_chromosome(length: int) -> list[int]:
    chromosome = [i for i in range(length)]
    random.shuffle(chromosome)

    return chromosome


def distance(t1: Town, t2: Town) -> float:
    return math.sqrt(math.pow(t1.x - t2.x, 2) + math.pow(t1.y - t2.y, 2))


def fitness(chromosome: list[int], towns: list[Town]) -> float:
    total_distance = 0
    for i in range(len(towns) - 1):
        total_distance += distance(towns[chromosome[i]], towns[chromosome[i + 1]])

    return 1 / total_distance


def tournament(population):
    selected = []
    indices = [i for i in range(len(population))]

    for _ in range(len(population) // 2):
        first, second = random.choices(indices, k=2)
        # while first == second:
        #     print("selection conflict")
        #     second = random.choice(indices)

        if population[first].fitness > population[second].fitness:
            selected.append(population[first])
        else:
            selected.append(population[second])

        indices.remove(first)
        try:
            indices.remove(second)
        except ValueError:
            pass

    return selected


def one_point_no_rep(father, mother) -> tuple:
    crossover_point = random.randint(1, len(father.chromosome) - 2)
    offspring1 = father.chromosome[:crossover_point]
    offspring2 = father.chromosome[crossover_point:]

    for gene in mother.chromosome:
        if gene not in offspring1:
            offspring1.append(gene)
        else:
            offspring2.append(gene)

    return offspring1, offspring2


def rotation(offspring):
    indices = [i for i in range(len(offspring.chromosome))]
    a, b = random.choices(indices, k=2)
    # while a == b:
    #     print("mutation conflict")
    #     b = random.choice(indices)
    first = a if a < b else b
    second = a if a > b else b

    head = offspring.chromosome[:first]
    middle = reversed(offspring.chromosome[first:second])
    tail = offspring.chromosome[second:]
    head.extend(middle)
    head.extend(tail)
    offspring.chromosome = head

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
    x = data["x"]
    y = data["y"]
    towns = [Town(x.iloc[i], y.iloc[i]) for i in range(len(data))]

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

    evaluator = PipeEvaluator(fitness, towns)
    crossoverer = PipeCrossover(one_point_no_rep)

    # generate initial population
    start = time.perf_counter()
    population = genetic.generate(N, generate_chromosome, len(towns))
    end = time.perf_counter()
    timings["generation"] += end - start

    start = time.perf_counter()
    # population = genetic.evaluation(population, fitness, towns)
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
        selected = genetic.selection(population, tournament)
        end = time.perf_counter()
        timings["selection"] += end - start

        # crossover
        start = time.perf_counter()
        # offsprings = genetic.crossover(selected, one_point_no_rep)
        offsprings = crossoverer.crossover(selected)
        end = time.perf_counter()
        timings["crossover"] += end - start

        # mutation
        start = time.perf_counter()
        offsprings = genetic.mutation(offsprings, rotation, mutation_rate)
        end = time.perf_counter()
        timings["mutation"] += end - start

        # offsprings evaluation
        start = time.perf_counter()
        # offsprings = genetic.evaluation(offsprings, fitness, towns)
        evaluator.evaluate(offsprings)
        end = time.perf_counter()
        timings["evaluation"] += end - start

        # replacement
        start = time.perf_counter()
        population = genetic.replace(population, offsprings, merge_replace)
        end = time.perf_counter()
        timings["replacement"] += end - start

        if best.fitness < population[0].fitness:
            best = population[0]

        best_fitness.append(best.fitness)
        # if best.fitness == average_fitness[-1]:
        #     print(f"stopped at generation: {g}")
        #     break

    print(f"best solution: {best.fitness}")
    evaluator.shutdown()
    crossoverer.shutdown()

    # drawing the graph
    plotting.draw_graph(towns, best)

    # plotting data
    plotting.fitness_trend(average_fitness, best_fitness)
    # print(biodiversities)
    plotting.biodiversity_trend(biodiversities)

    # timing
    plotting.timing(timings)

    for k in timings.keys():
        print(f"{k}: {timings[k]:.3f} seconds")
    print(f"total time: {sum(timings.values()):.3f} seconds")
