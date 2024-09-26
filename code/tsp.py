import random

import numpy as np
import pandas as pd


def generate(length: int) -> np.ndarray:
    chromosome = [i for i in range(length)]
    random.shuffle(chromosome)

    return np.array(chromosome)


def distance(t1, t2) -> float:
    return np.sqrt(np.pow(t1[0] - t2[0], 2) + np.pow(t1[1] - t2[1], 2))


def fitness(towns: np.ndarray, chromosome: np.ndarray):
    total_distance = 0.0
    for i in range(len(chromosome) - 1):
        total_distance += distance(towns[chromosome[i]], towns[chromosome[i + 1]])

    return 1.0 / total_distance


def tournament(scores: np.ndarray) -> list[int]:
    selected = []
    indices = [i for i in range(len(scores))]

    for _ in range(len(scores) // 2):
        first, second = random.choices(indices, k=2)
        while first == second:
            first, second = random.choices(indices, k=2)

        if scores[first] > scores[second]:
            selected.append(first)
            indices.remove(first)
        else:
            selected.append(second)
            indices.remove(second)

    return selected


def one_point_no_rep(father: np.ndarray, mother: np.ndarray) -> tuple:
    crossover_point = random.randint(1, len(father) - 2)

    offspring1 = father[:crossover_point]
    offspring2 = father[crossover_point:]

    tail1 = mother[np.isin(mother, offspring2)]
    tail2 = mother[np.isin(mother, offspring1)]

    return np.append(offspring1, tail1), np.append(offspring2, tail2)


def rotation(offspring: np.ndarray) -> np.ndarray:
    a = random.randint(0, len(offspring) - 1)
    b = random.randint(0, len(offspring) - 1)

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

    return np.array(next_generation), np.array(next_gen_scores)
