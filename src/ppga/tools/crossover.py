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


def two_points(father, mother) -> tuple:
    cx_point1 = random.randint(1, len(father) - 2)
    cx_point2 = random.randint(1, len(father) - 2)
    while cx_point2 == cx_point1:
        cx_point2 = random.randint(1, len(father) - 2)

    if cx_point1 > cx_point2:
        cx_point1, cx_point2 = cx_point2, cx_point1

    offspring1 = father[:cx_point1] + mother[cx_point1:cx_point2] + father[cx_point2:]
    offspring2 = mother[:cx_point1] + father[cx_point1:cx_point2] + mother[cx_point2:]

    return offspring1, offspring2


def shuffle(father, mother) -> tuple:
    offspring1 = []
    offspring2 = []
    for i in range(len(father)):
        if random.random() < 0.5:
            offspring1.append(father[i])
            offspring2.append(mother[i])
        else:
            offspring1.append(mother[i])
            offspring2.append(father[i])

    return offspring1, offspring2
