from ppga.base import Individual


def partial(
    population: list[Individual], offsprings: list[Individual], keep: float = 0.5
) -> list[Individual]:
    """Try to keep the specified percentage of the old generation"""
    if keep == 0.0:
        return offsprings

    n = round(len(population) * keep)
    to_keep = sorted(population, reverse=True)[:n]

    return sorted(to_keep + offsprings, reverse=True)[: len(population)]


def total(
    population: list[Individual], offsprings: list[Individual]
) -> list[Individual]:
    """
    Replace completely the old population with the new individuals.
    Equal to call the `partial` replacement with `keep` parameter equal to 0.0.
    """
    return offsprings


def elitist(
    population: list[Individual], offsprings: list[Individual]
) -> list[Individual]:
    """
    The new population contains the best individuals of both old and new generation.
    It calls the `partial` replacement with `keep` parameter equal to 1.0.
    """
    return partial(population, offsprings, 1.0)
