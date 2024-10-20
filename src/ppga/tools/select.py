import random

from ppga.base.individual import Individual


def tournament(population: list[Individual], tournsize: int = 2) -> list[Individual]:
    """
    Tournament selection where `tournsize` individuals clash to be selected until
    a number equal to the half of the population size is reached.

    Args:
        population: a list of individuals.
        tournsize: how many individuals clash.

    Returns:
        the selected individuals.
    """
    selected = []

    for _ in range(len(population) // 2):
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
            population, k=len(population) // 2, weights=normalized_scores
        )
    except ZeroDivisionError:
        selected = population[: len(population) // 2]

    return selected
