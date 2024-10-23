import random

from ppga.base import Individual


def couples_mating(population: list[Individual]) -> list[tuple]:
    indices = [i for i in range(len(population))]
    couples = []
    for _ in range(len(population) // 2):
        father, mother = random.sample(indices, k=2)
        couples.append((population[father], population[mother]))
        indices.remove(father)
        indices.remove(mother)

    return couples
