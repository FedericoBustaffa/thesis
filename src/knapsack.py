import random
import sys

from ppga import algorithms, base, log, tools
from utils import plotting


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
    if len(argv) < 4:
        print(f"USAGE: py {argv[0]} <Items> <N> <G> <LOG_LEVEL>")
        exit(1)

    if len(argv) < 5:
        argv.append("INFO")
    logger = log.getLogger()

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
    toolbox.set_generation(tools.gen_repetition, (0, 1), len(items))
    toolbox.set_selection(tools.sel_roulette)
    toolbox.set_crossover(tools.cx_uniform)
    toolbox.set_mutation(tools.mut_bitflip)
    toolbox.set_evaluation(evaluate, items, capacity)
    toolbox.set_replacement(tools.merge)

    hof = base.HallOfFame(10)

    # sequential execution
    best, stats = algorithms.sga(toolbox, N, 0.8, 0.2, G, hall_of_fame=hof)

    value, weight = show_solution(best[0].chromosome, items)
    logger.info(f"sequential best solution: ({value:.3f}, {weight:.3f})")
    logger.info(f"sequential best fitnes: {best[0].fitness}")
    for i, ind in enumerate(hof):
        logger.debug(f"{i}. {ind.values}")

    # parallel execution
    hof.clear()
    pbest, pstats = algorithms.psga(toolbox, N, 0.8, 0.2, G, hall_of_fame=hof)

    value, weight = show_solution(pbest[0].chromosome, items)
    logger.info(f"queue best solution: ({value:.3f}, {weight:.3f})")
    logger.info(f"queue best fitness: {pbest[0].fitness}")
    for i, ind in enumerate(hof):
        logger.debug(f"{i}. {ind.values}")

    # plotting
    if logger.level <= log.SUCCESS:
        plotting.fitness_trend(stats)
        plotting.biodiversity_trend(stats)

        plotting.fitness_trend(pstats)
        plotting.biodiversity_trend(pstats)


if __name__ == "__main__":
    main(sys.argv)
