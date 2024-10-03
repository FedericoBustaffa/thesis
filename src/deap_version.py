import random
import sys
from functools import partial

import numpy as np
import pandas as pd
from deap import base, creator, tools
from loguru import logger

import tsp


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
    toolbox.population(n=N)

    evaluate = partial(tsp.fitness, towns)
    toolbox.register("evaluate", evaluate)

    toolbox.register("select", tools.selTournament, tournsize=2)
    ind1 = toolbox.individual()
    ind2 = toolbox.individual()

    logger.debug(f"ind1: {ind1}")
    logger.debug(f"ind2: {ind2}")

    tools.cxOrdered(ind1, ind2)
    logger.debug(f"ind1: {ind1}")
    logger.debug(f"ind2: {ind2}")


if __name__ == "__main__":
    main()
