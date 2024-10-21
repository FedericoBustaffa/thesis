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


def bit_flip(chromosome, pb: float = 0.5):
    for gene in chromosome:
        if random.random() < pb:
            gene = not gene

    return chromosome


def shuffle(chromosome, pb: float = 0.5):
    for i, gene in enumerate(chromosome):
        if random.random() < pb:
            new_pos = random.randint(0, len(chromosome) - 1)
            while new_pos == i:
                new_pos = random.randint(0, len(chromosome) - 1)
            chromosome[i], chromosome[new_pos] = chromosome[new_pos], chromosome[i]

    return chromosome
