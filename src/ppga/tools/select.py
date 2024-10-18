import random

from ppga.base.individual import Individual


def tournament(population: list[Individual], tournsize: int = 2) -> list[Individual]:
    selected = []

    for _ in range(len(population) // 2):
        clash = random.choices(population, k=tournsize)
        winner = max(clash)
        selected.append(winner)

    return selected
