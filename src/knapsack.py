import random
import sys
import time

from loguru import logger

from ppga import base
from ppga.algorithms import parallel, sequential
from ppga.tools import crossover, generation, mutation, replacement, selection


class Item:
    def __init__(self, value: float, weight: float):
        self.value = value
        self.weight = weight


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
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | {file}:{line} | <level>{level} - {message}</level>",
        level="TRACE",
        enqueue=True,
    )

    if len(argv) != 4:
        print(f"USAGE: py {argv[0]} <Items> <N> <G>")
        exit(1)

    items_num = int(argv[1])
    N = int(argv[2])
    G = int(argv[3])

    items = [Item(random.random(), random.random()) for _ in range(items_num)]
    capacity = sum([i.weight for i in items]) * 0.7
    logger.info(f"capacity: {capacity:.3f}")

    # greedy method for comparison
    solution, items = greedy(items, capacity)
    value, weight = show_solution(solution, items)
    logger.info(f"greedy (value: {value:.3f}, weight: {weight:.3f})")

    toolbox = base.ToolBox()
    toolbox.set_weights(weights=(3.0, -1.0))
    toolbox.set_attributes(random.randint, 0, 1)
    toolbox.set_generation(generation.repeat, len(items))
    toolbox.set_selection(selection.roulette)
    toolbox.set_crossover(crossover.shuffle, cxpb=0.8)
    toolbox.set_mutation(mutation.shuffle, mutpb=0.2)
    toolbox.set_evaluation(evaluate, items, capacity)
    toolbox.set_replacement(replacement.merge)

    hof = base.HallOfFame(10)

    start = time.perf_counter()
    seq_best, seq_stats = sequential.generational(toolbox, N, G, hall_of_fame=hof)
    sequential_time = time.perf_counter() - start

    logger.success(f"sequential time: {sequential_time} seconds")
    value, weight = show_solution(seq_best[0].chromosome, items)
    logger.success(f"sequential best solution: ({value:.3f}, {weight:.3f})")
    logger.success(f"sequential best fitnes: {seq_best[0].fitness}")
    print(f"Hall of Fame\n{hof}")

    hof.clear()
    start = time.perf_counter()
    queue_best, queue_stats = parallel.generational(toolbox, N, G, hall_of_fame=hof)
    queue_time = time.perf_counter() - start

    print(f"Hall of Fame\n{hof}")

    logger.success(f"queue time: {queue_time} seconds")
    value, weight = show_solution(queue_best[0].chromosome, items)
    logger.success(f"queue best solution: ({value:.3f}, {weight:.3f})")
    logger.success(f"queue best fitness: {queue_best[0].fitness}")

    speed_up = sequential_time / queue_time
    true_speed_up = seq_stats.cme() / queue_stats["parallel"]
    if speed_up < 1.0:
        logger.warning(f"speed up: {speed_up} seconds")
    else:
        logger.success(f"speed up: {speed_up} seconds")

    if true_speed_up < 1.0:
        logger.warning(f"true speed up: {true_speed_up} seconds")
    else:
        logger.success(f"true speed up: {true_speed_up} seconds")


if __name__ == "__main__":
    main(sys.argv)
