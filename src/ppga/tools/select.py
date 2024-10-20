import random

from ppga.base.individual import Individual


def tournament(population: list[Individual], tournsize: int = 2) -> list[Individual]:
    selected = []

    for _ in population:
        clash = random.choices(population, k=tournsize)
        winner = max(clash)
        selected.append(winner)

    return selected


def roulette(population: list[Individual]) -> list[Individual]:
    selected = []
    total_fitness = sum([i.fitness for i in population])
    try:
        normalized_scores = [i.fitness / total_fitness for i in population]
        selected = random.choices(
            population, k=len(population), weights=normalized_scores
        )
    except ZeroDivisionError:
        selected = population[: len(population) // 2]

    return selected
