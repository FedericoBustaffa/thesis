import random


def one_point(father, mother) -> tuple:
    crossover_point = random.randint(1, len(father) - 2)

    offspring1 = father[:crossover_point] + mother[crossover_point:]
    offspring2 = father[crossover_point:] + mother[:crossover_point]

    return offspring1, offspring2


def one_point_ordered(father, mother) -> tuple:
    crossover_point = random.randint(1, len(father) - 2)

    offspring1 = father[:crossover_point]
    offspring2 = father[crossover_point:]

    for gene in mother:
        if gene not in offspring1:
            offspring1.append(gene)
        else:
            offspring2.append(gene)

    return offspring1, offspring2
