import random


def cx_one_point(father, mother) -> tuple:
    crossover_point = random.randint(1, len(father) - 2)

    offspring1 = father[:crossover_point] + mother[crossover_point:]
    offspring2 = mother[:crossover_point] + father[crossover_point:]

    return offspring1, offspring2


def cx_one_point_ordered(father, mother) -> tuple:
    crossover_point = random.randint(1, len(father) - 2)

    offspring1 = father[:crossover_point]
    offspring2 = father[crossover_point:]

    offspring1 += [gene for gene in mother if gene not in offspring1]
    offspring2 += [gene for gene in mother if gene not in offspring2]

    return offspring1, offspring2


def cx_two_points(father, mother) -> tuple:
    cx_point1, cx_point2 = random.sample([i + 1 for i in range(len(father) - 2)], k=2)

    if cx_point1 > cx_point2:
        cx_point1, cx_point2 = cx_point2, cx_point1

    offspring1 = father[:cx_point1] + mother[cx_point1:cx_point2] + father[cx_point2:]
    offspring2 = mother[:cx_point1] + father[cx_point1:cx_point2] + mother[cx_point2:]

    return offspring1, offspring2


def cx_uniform(father, mother, indpb: float = 0.5) -> tuple:
    assert indpb >= 0.0 and indpb <= 1.0

    offspring1 = []
    offspring2 = []
    for i in range(len(father)):
        if random.random() < indpb:
            offspring1.append(mother[i])
            offspring2.append(father[i])
        else:
            offspring1.append(father[i])
            offspring2.append(mother[i])

    return offspring1, offspring2


if __name__ == "__main__":
    father = random.sample([i for i in range(10)], k=10)
    mother = random.sample([i for i in range(10)], k=10)

    print(father, mother)

    o1, o2 = cx_uniform(father, mother)
    print(o1, o2)
