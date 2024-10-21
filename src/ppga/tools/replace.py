from ppga.base import Individual


def total(
    population: list[Individual], offsprings: list[Individual]
) -> list[Individual]:
    next_generation = sorted(offsprings, reverse=True)
    population = sorted(population)
    next_generation.extend(population[len(offsprings) : len(population)])

    return next_generation


def merge(
    population: list[Individual], offsprings: list[Individual]
) -> list[Individual]:
    next_generation = sorted(population + offsprings, reverse=True)

    return next_generation[: len(population)]
