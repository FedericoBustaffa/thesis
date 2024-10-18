from ppga.base import Individual


def merge(
    population: list[Individual], offsprings: list[Individual]
) -> list[Individual]:
    population = sorted(population, reverse=True)
    offsprings = sorted(offsprings, reverse=True)

    next_generation = []
    index = 0
    index1 = 0
    index2 = 0

    while (
        index < len(population)
        and index1 < len(population)
        and index2 < len(offsprings)
    ):
        if population[index1] > offsprings[index2]:
            next_generation.append(population[index1])
            index1 += 1
        else:
            next_generation.append(offsprings[index2])
            index2 += 1
        index += 1

    if index1 >= len(population):
        return next_generation
    elif index2 >= len(offsprings):
        next_generation[index:] = population[index1 : len(population) - index2]

    return next_generation
