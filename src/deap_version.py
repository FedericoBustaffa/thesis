import random
import sys
from functools import partial

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from loguru import logger

import tsp
from utils import plotting


def main():
    if len(sys.argv) < 7:
        logger.error(f"USAGE: py {sys.argv[0]} <T> <N> <G> <C> <M> <log_level>")
        exit(1)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | {file}:{line} | <level>{level} - {message}</level>",
        level=sys.argv[6].upper(),
        enqueue=True,
    )

    data = pd.read_csv(f"datasets/towns_{sys.argv[1]}.csv")
    towns = np.array([[data["x"].iloc[i], data["y"].iloc[i]] for i in range(len(data))])

    # Initial population size
    N = int(sys.argv[2])

    # Max generations
    G = int(sys.argv[3])

    # crossover rate
    CR = float(sys.argv[4])

    # mutation rate
    MR = float(sys.argv[5])

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(towns)), len(towns))
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.indices
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=N)

    evaluate = partial(tsp.fitness, towns)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutInversion)
    hof = tools.HallOfFame(5)

    result, logbook = algorithms.eaSimple(
        population=pop,
        toolbox=toolbox,
        cxpb=CR,
        mutpb=MR,
        ngen=G,
        halloffame=hof,
        verbose=True,
    )
    logger.debug(f"best solution: {type(result)}")
    plotting.draw_graph(data, hof.items[-1])


if __name__ == "__main__":
    main()
