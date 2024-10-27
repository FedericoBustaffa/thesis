import math
import sys

import pandas as pd

from ppga import algorithms, base, log, tools
from utils import plotting


class Town:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def distance(t1: Town, t2: Town) -> float:
    return math.sqrt(math.pow(t1.x - t2.x, 2) + math.pow(t1.y - t2.y, 2))


def evaluate(chromosome, towns: list[Town]) -> tuple[float]:
    total_distance = 0.0
    for i in range(len(chromosome) - 1):
        total_distance += distance(towns[chromosome[i]], towns[chromosome[i + 1]])

    return (total_distance,)


def main(argv: list[str]):
    if len(argv) < 4:
        print(f"USAGE: py {argv[0]} <T> <N> <G> <LOG-LEVEL>")
        exit(1)

    if len(argv) < 5:
        argv.append("INFO")
    logger = log.getLogger("main", argv[-1].upper())

    data = pd.read_csv(f"datasets/towns_{argv[1]}.csv")
    x_coords = data["x"]
    y_coords = data["y"]
    towns = [Town(x, y) for x, y in zip(x_coords, y_coords)]

    # Initial population size
    N = int(argv[2])

    # Max generations
    G = int(argv[3])

    toolbox = base.ToolBox()
    toolbox.set_weights((-1.0,))
    toolbox.set_generation(tools.gen_permutation, range(len(towns)))
    toolbox.set_selection(tools.sel_tournament, tournsize=2)
    toolbox.set_crossover(tools.cx_one_point_ordered)
    toolbox.set_mutation(tools.rotation)
    toolbox.set_evaluation(evaluate, towns)
    toolbox.set_replacement(tools.merge)

    hall_of_fame = base.HallOfFame(5)

    # sequential execution
    best, stats = algorithms.sga(toolbox, N, 0.7, 0.3, G, hall_of_fame)
    logger.info(f"sequential best score: {best[0].fitness}")
    for i, ind in enumerate(hall_of_fame):
        logger.debug(f"{i + 1}. {ind.values}")

    # parallel execution
    hall_of_fame.clear()
    pbest, pstats = algorithms.psga(toolbox, N, 0.7, 0.3, G, hall_of_fame)
    logger.info(f"parallel best score: {pbest[0].fitness}")
    for i, ind in enumerate(hall_of_fame):
        logger.debug(f"{i + 1}. {ind.values}")

    # statistics data
    if logger.level <= log.SUCCESS:
        plotting.draw_graph(data, best[0].chromosome)
        plotting.fitness_trend(stats)
        plotting.biodiversity_trend(stats)

        plotting.draw_graph(data, pbest[0].chromosome)
        plotting.fitness_trend(pstats)
        plotting.biodiversity_trend(pstats)

        plotting.evals(stats.evals)
        plotting.multievals(pstats.multievals)


if __name__ == "__main__":
    main(sys.argv)
