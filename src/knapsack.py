import random
import sys
import time

from ppga import base
from ppga.algorithms import parallel, sequential
from ppga.tools import crossover, mutation, replacement, selection


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
    if len(argv) < 4:
        print(f"USAGE: py {argv[0]} <Items> <N> <G> <C> <M> <W>")
        exit(1)

    items_num = int(argv[1])
    N = int(argv[2])
    G = int(argv[3])
    W = int(argv[4])

    items = [Item(random.random(), random.random()) for _ in range(items_num)]
    capacity = sum([i.weight for i in items]) * 0.7
    print(f"capacity: {capacity:.3f}")

    # greedy method for comparison
    solution, items = greedy(items, capacity)
    value, weight = show_solution(solution, items)
    print(f"greedy (value: {value:.3f}, weight: {weight:.3f})")

    toolbox = base.ToolBox()
    toolbox.set_weights(weights=(3.0, -1.0))
    toolbox.set_generation(generate, len(items))
    toolbox.set_selection(selection.roulette)
    toolbox.set_crossover(crossover.shuffle, cxpb=0.8)
    toolbox.set_mutation(mutation.shuffle, mutpb=0.2)
    toolbox.set_evaluation(evaluate, items, capacity)
    toolbox.set_replacement(replacement.merge)

    start = time.perf_counter()
    seq_best, seq_stats = sequential.generational(toolbox, N, G)
    sequential_time = time.perf_counter() - start
    print(f"sequential time: {sequential_time} seconds")
    value, weight = show_solution(seq_best[0].chromosome, items)
    print(f"sequential best solution: ({value:.3f}, {weight:.3f})")
    print(f"sequential best fitnes: {seq_best[0].fitness}")

    start = time.perf_counter()
    queue_best, queue_stats = parallel.generational(toolbox, N, G, W)
    queue_time = time.perf_counter() - start
    print(f"queue time: {queue_time} seconds")
    value, weight = show_solution(queue_best[0].chromosome, items)
    print(f"queue best solution: ({value:.3f}, {weight:.3f})")

    print(f"speed up: {sequential_time / queue_time:.4f} seconds")


if __name__ == "__main__":
    main(sys.argv)
