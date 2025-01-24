import random
import sys
import time
from functools import partial

import pandas as pd
from common import Town, evaluate
from scoop import futures
from utils import plotting

from deap import algorithms, base, creator, tools


def main(argv: list[str]):
    if len(argv) < 4:
        print(f"USAGE: py {argv[0]} <T> <N> <G>")
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
