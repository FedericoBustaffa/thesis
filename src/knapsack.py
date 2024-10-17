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


def evaluate(chromosome, items: list[Item], capacity: float) -> tuple:
    value = 0.0
    weight = 0.0

    for i in range(len(items)):
        value += chromosome[i] * items[i].value
        weight += chromosome[i] * items[i].weight

    if weight > capacity:
        return -value, weight
    else:
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

    return chromosome


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


def greedy(items: list[Item], capacity: float) -> tuple[list[int], list[Item]]:
    items = sorted(items, key=lambda x: x.value / x.weight, reverse=True)
    weight = 0.0

    solution = []
    for i in items:
        if i.weight + weight <= capacity:
            solution.append(1)
            weight += i.weight
        else:
            solution.append(0)

    return solution, items


def show_solution(solution, items):
    value = sum([i.value * s for i, s in zip(items, solution)])
    weight = sum([i.weight * s for i, s in zip(items, solution)])

    return value, weight


def main(argv: list[str]):
    if len(argv) < 6:
        print(f"USAGE: py {argv[0]} <Items> <N> <G> <C> <M> <W>")
        exit(1)

    items_num = int(argv[1])
    N = int(argv[2])
    G = int(argv[3])
    CP = float(argv[4])
    MP = float(argv[5])
    W = int(argv[6])

    items = [Item(random.random(), random.random()) for _ in range(items_num)]
    capacity = sum([i.weight for i in items]) * 0.7
    print(f"capacity: {capacity:.3f}")

    # greedy method for comparison
    solution, items = greedy(items, capacity)
    value, weight = show_solution(solution, items)
    print(f"greedy (value: {value:.3f}, weight: {weight:.3f})")

    toolbox = base.ToolBox()
    toolbox.set_fitness_weights(weights=(2.0, -0.5))
    toolbox.set_generation(generate, len(items))
    toolbox.set_selection(select)
    toolbox.set_mating(mate)
    toolbox.set_crossover(cx_one_point, CP)
    toolbox.set_mutation(mut_bitswap, MP)
    toolbox.set_evaluation(evaluate, items, capacity)
    toolbox.set_replacement(merge)

    genetic_solver = solver.GeneticSolver()
    start = time.perf_counter()
    seq_best, seq_stats = genetic_solver.run(toolbox, N, G, base.Statistics())
    sequential_time = time.perf_counter() - start
    print(f"sequential time: {sequential_time} seconds")
    value, weight = show_solution(seq_best[0].chromosome, items)
    print(f"sequential best solution: ({value:.3f}, {weight:.3f})")

    queue_solver = solver.QueuedGeneticSolver(W)
    start = time.perf_counter()
    queue_best, queue_stats = queue_solver.run(toolbox, N, G, base.Statistics())
    queue_time = time.perf_counter() - start
    print(f"queue time: {queue_time} seconds")
    value, weight = show_solution(queue_best[0].chromosome, items)
    print(f"sequential best solution: ({value:.3f}, {weight:.3f})")

    print(f"speed up: {sequential_time / queue_time:.4f} seconds")


if __name__ == "__main__":
    main(sys.argv)
