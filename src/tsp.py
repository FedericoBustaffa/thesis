import math
import random


def generate(length: int):
    chromosome = [i for i in range(length)]
    random.shuffle(chromosome)

    return chromosome


def distance(t1, t2) -> float:
    return math.sqrt(math.pow(t1[0] - t2[0], 2) + math.pow(t1[1] - t2[1], 2))


def fitness(towns, chromosome):
    total_distance = 0.0
    for i in range(len(chromosome) - 1):
        total_distance += distance(towns[chromosome[i]], towns[chromosome[i + 1]])

    return 1.0 / total_distance


def tournament(population, scores):
    selected = []
    indices = [i for i in range(len(scores))]

    for _ in range(len(scores) // 2):
        first, second = random.choices(indices, k=2)
        while first == second:
            first, second = random.choices(indices, k=2)

        if scores[first] > scores[second]:
            selected.append(population[first])
            indices.remove(first)
        else:
            selected.append(population[second])
            indices.remove(second)

    return selected


def couples_mating(chosen):
    indices = [i for i in range(len(chosen))]
    couples = []
    for _ in range(len(chosen) // 2):
        father, mother = random.sample(indices, k=2)
        couples.append((chosen[father], chosen[mother]))
        indices.remove(father)
        indices.remove(mother)

    return couples


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
    a = random.randint(0, len(offspring) - 1)
    b = random.randint(0, len(offspring) - 1)

    while a == b:
        b = random.randint(0, len(offspring) - 1)

    first = a if a < b else b
    second = a if a > b else b
    offspring[first:second] = reversed(offspring[first:second])

    return offspring


def merge_replace(population, scores1, offsprings, scores2) -> tuple:
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
        next_generation[index:] = population[index1 : len(population) - index2]
        next_gen_scores[index:] = scores1[index1 : len(scores1) - index2]

    return next_generation, next_gen_scores
