from ppga.base import Individual


def total(
    population: list[Individual], offsprings: list[Individual]
) -> list[Individual]:
    return offsprings


def merge(
    population: list[Individual], offsprings: list[Individual]
) -> list[Individual]:
    return sorted(population + offsprings, reverse=True)[: len(population)]
