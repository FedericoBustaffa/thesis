import math
import random
import sys
import time
from functools import partial

import pandas as pd
from deap import algorithms, base, creator, tools
from scoop import futures

from utils import plotting


class Town:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def distance(t1: Town, t2: Town) -> float:
    return math.sqrt(math.pow(t1.x - t2.x, 2) + math.pow(t1.y - t2.y, 2))


def evaluate(towns: list[Town], chromosome) -> tuple[float]:
    total_distance = 0.0
    for i in range(len(chromosome) - 1):
        total_distance += distance(towns[chromosome[i]], towns[chromosome[i + 1]])

    return (total_distance,)


def main(argv: list[str]):
    if len(argv) < 4:
        print(f"USAGE: py {argv[0]} <T> <N> <G> <W>")
        exit(1)

    data = pd.read_csv(f"datasets/towns_{argv[1]}.csv")
    x_coords = data["x"]
    y_coords = data["y"]
    towns = [Town(x, y) for x, y in zip(x_coords, y_coords)]

    L = int(argv[1])

    # Initial population size
    N = int(argv[2])

    # Max generations
    G = int(argv[3])

    # crossover rate
    cxpb = 0.7

    # mutation rate
    mutpb = 0.3

    # number of workers
    W = int(argv[4])

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(L), L)
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.indices
    )

    hall_of_fame = tools.HallOfFame(5)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", partial(evaluate, towns))
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.01)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("map", futures.map)

    start = time.perf_counter()
    best, logbook = algorithms.eaSimple(
        toolbox.population(n=N),
        toolbox=toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=G,
        halloffame=hall_of_fame,
    )
    end = time.perf_counter()

    for i in best:
        print(f"{i.fitness}")

    print("HOF")
    for i in hall_of_fame:
        print(f"{i.fitness}")

    print(f"DEAP time: {end - start:.4f} seconds")
    plotting.draw_graph(
        data, sorted([ind for ind in best], key=lambda x: x.fitness, reverse=True)[0]
    )
    plotting.draw_graph(data, hall_of_fame[0])


if __name__ == "__main__":
    main(sys.argv)
