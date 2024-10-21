from ppga.base import Individual


def total(
    population: list[Individual], offsprings: list[Individual]
) -> list[Individual]:
    return offsprings


def merge(
    population: list[Individual], offsprings: list[Individual]
) -> list[Individual]:
    next_generation = sorted(population + offsprings, reverse=True)

    return next_generation[: len(population)]
