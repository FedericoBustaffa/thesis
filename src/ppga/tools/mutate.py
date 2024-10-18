import random


def rotation(chromosome):
    a = random.randint(0, len(chromosome) - 1)
    b = random.randint(0, len(chromosome) - 1)

    while a == b:
        b = random.randint(0, len(chromosome) - 1)

    first = a if a < b else b
    second = a if a > b else b
    chromosome[first:second] = reversed(chromosome[first:second])

    return chromosome


def bitswap(chromosome):
    position = random.randint(0, len(chromosome) - 1)
    if chromosome[position] == 0:
        chromosome[position] = 1
    else:
        chromosome[position] = 0

    return chromosome
