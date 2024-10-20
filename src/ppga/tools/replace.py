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
    population = sorted(population, reverse=True)
    offsprings = sorted(offsprings, reverse=True)

    next_generation = []
    index1 = 0
    index2 = 0
    for index in range(len(population)):
        if population[index1] > offsprings[index2]:
            next_generation.append(population[index1])
            index1 += 1
        else:
            next_generation.append(offsprings[index2])
            index2 += 1

    return next_generation
