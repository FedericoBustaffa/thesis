import random


def mut_bitflip(chromosome, indpb: float = 0.5):
    attr_type = type(chromosome[0])
    for i, gene in enumerate(chromosome):
        if random.random() < indpb:
            chromosome[i] = attr_type(not gene)

    return chromosome


def mut_swap(chromosome, pb: float = 0.5):
    for i, gene in enumerate(chromosome):
        if random.random() < pb:
            new_pos = random.randint(0, len(chromosome) - 1)
            while new_pos == i:
                new_pos = random.randint(0, len(chromosome) - 1)
            chromosome[i], chromosome[new_pos] = chromosome[new_pos], chromosome[i]

    return chromosome


def mut_rotation(chromosome):
    a, b = random.sample([i for i in range(len(chromosome) + 1)], k=2)

    first = a if a < b else b
    second = a if a > b else b
    chromosome[first:second] = reversed(chromosome[first:second])

    return chromosome


if __name__ == "__main__":
    chromosome = [random.randint(0, 1) for _ in range(10)]
    print(chromosome)
    print(mut_bitflip(chromosome))
