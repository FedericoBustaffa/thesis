import random

from ppga.base.individual import Individual


def tournament(
    population: list[Individual], population_size: int, tournsize: int = 2
) -> list[Individual]:
    selected = []

    for _ in range(population_size):
        clash = random.choices(population, k=tournsize)
        winner = max(clash)
        selected.append(winner)

    return selected


def roulette(population: list[Individual], population_size: int) -> list[Individual]:
    total = 0.0
    for i in population:
        if i.fitness < 0.0:
            total -= 1.0 / i.fitness
        else:
            total += i.fitness

    if total == 0.0:
        return random.choices(population, k=population_size)
    else:
        normalized_scores = []
        for i in population:
            if i.fitness < 0.0:
                normalized_scores.append(-1.0 / i.fitness / total)
            else:
                normalized_scores.append(i.fitness / total)
        return random.choices(population, k=population_size, weights=normalized_scores)
