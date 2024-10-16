import random
import sys
import time

from ppga import base, solver


class Item:
    def __init__(self, value: float, weight: float):
        self.value = value
        self.weight = weight


def generate(length: int) -> list[int]:
    return [random.randint(0, 1) for _ in range(length)]


def evaluate(chromosome, items: list[Item]) -> tuple:
    value = 0.0
    weight = 0.0

    for i in range(len(items)):
        value += chromosome[i] * items[i].value
        weight += chromosome[i] * items[i].weight

    return value, weight


def select(population: list[base.Individual]) -> list[base.Individual]:
    selected = []
    indices = [i for i in range(len(population))]

    for _ in range(len(population) // 2):
        first, second = random.choices(indices, k=2)
        while first == second:
            first, second = random.choices(indices, k=2)

        if population[first] > population[second]:
            selected.append(population[first])
            indices.remove(first)
        else:
            selected.append(population[second])
            indices.remove(second)

    return selected


def mate(population: list[base.Individual]) -> list[tuple]:
    indices = [i for i in range(len(population))]
    couples = []
    for _ in range(len(population) // 2):
        father, mother = random.sample(indices, k=2)
        couples.append((population[father], population[mother]))
        indices.remove(father)
        indices.remove(mother)

    return couples


def cx_one_point(father: list[int], mother: list[int]) -> tuple:
    crossover_point = random.randint(1, len(father) - 2)

    offspring1 = father[:crossover_point] + mother[crossover_point:]
    offspring2 = father[crossover_point:] + mother[:crossover_point]

    return offspring1, offspring2


def mut_bitswap(chromosome: list[int]) -> list[int]:
    position = random.randint(0, len(chromosome) - 1)
    if chromosome[position] == 0:
        chromosome[position] = 1
    else:
        chromosome[position] = 0


def merge(
    population: list[base.Individual], offsprings: list[base.Individual]
) -> list[base.Individual]:
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


def main(argv: list[int]):
    if len(argv) < 6:
        print(f"USAGE: py {argv[0]} <Items> <N> <G> <C> <M> <W>")
        exit(1)

    Items = int(argv[1])
    N = int(argv[2])
    G = int(argv[3])
    CP = float(argv[4])
    MP = float(argv[5])
    W = int(argv[6])

    items = [Item(random.random(), random.random()) for _ in range(Items)]

    toolbox = base.ToolBox()
    toolbox.set_fitness_weights(weights=(2.0, 1.0))
    toolbox.set_generation(generate, len(items))
    toolbox.set_selection(select)
    toolbox.set_mating(mate)
    toolbox.set_crossover(cx_one_point, CP)
    toolbox.set_mutation(mut_bitswap, MP)
    toolbox.set_evaluation(evaluate, items)
    toolbox.set_replacement(merge)

    genetic_solver = solver.GeneticSolver()
    start = time.perf_counter()
    seq_best, seq_stats = genetic_solver.run(toolbox, N, G, base.Statistics())
    sequential_time = time.perf_counter() - start
    print(f"sequential time: {sequential_time} seconds")


if __name__ == "__main__":
    main(sys.argv)
