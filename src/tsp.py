import math
import random
import sys
import time

import pandas as pd
from loguru import logger

from genetic_solver import GeneticSolver
from modules import Individual
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

    time.sleep(0.00005)

    return (total_distance,)


def tournament(population: list[Individual]):
    selected = []
    indices = [i for i in range(len(population))]

    for _ in range(len(population) // 2):
        first, second = random.choices(indices, k=2)
        while first == second:
            first, second = random.choices(indices, k=2)

        if population[first] < population[second]:
            selected.append(population[first])
            indices.remove(first)
        else:
            selected.append(population[second])
            indices.remove(second)

    return selected


def couples_mating(population: list[Individual]) -> list[tuple]:
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
    population: list[Individual], offsprings: list[Individual]
) -> list[Individual]:

    population = sorted(population)
    offsprings = sorted(offsprings)

    next_generation = []
    index = 0
    index1 = 0
    index2 = 0

    while (
        index < len(population)
        and index1 < len(population)
        and index2 < len(offsprings)
    ):
        if population[index1] < offsprings[index2]:
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
        logger.error(f"USAGE: py {sys.argv[0]} <T> <N> <G> <C> <M> <log_level>")
        exit(1)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | {file}:{line} | <level>{level} - {message}</level>",
        level=sys.argv[6].upper(),
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

    solver = GeneticSolver()
    solver.create_individual(attribute_type=int, structure=list)
    solver.create_fitness(weights=(-1.0,))
    solver.set_population(list)

    solver.set_generation(generate, len(towns))
    solver.set_selection(tournament)
    solver.set_mating(couples_mating)
    solver.set_crossover(cx_one_point_ordered, CR)
    solver.set_mutation(mut_rotation, MR)
    solver.set_evaluation(evaluate, towns)
    solver.set_replacement(merge)

    start = time.perf_counter()
    best = solver.run(N, G)
    logger.info(f"total time: {time.perf_counter() - start:.6f} seconds")

    logger.info(f"best score: {best[0]}")

    # # drawing the graph
    plotting.draw_graph(data, best[0].chromosome)

    # # statistics data
    # plotting.fitness_trend(ga.average_fitness, ga.best_fitness)
    # plotting.biodiversity_trend(ga.biodiversity)

    # # timing
    # plotting.timing(ga.timings)

    # for k in ga.timings.keys():
    #     print(f"{k}: {ga.timings[k]:.3f} seconds")
    # print(f"total time: {sum(ga.timings.values()):.3f} seconds")
    # print(f"total time: {sum(ga.timings.values()):.3f} seconds")
    # print(f"total time: {sum(ga.timings.values()):.3f} seconds")
    # print(f"total time: {sum(ga.timings.values()):.3f} seconds")
