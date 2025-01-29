import random

import numpy as np

from deap import algorithms, base, creator, tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def generate():
    return creator.Individual([random.uniform(-5, 5), random.uniform(-5, 5)])


def evaluate(chromosome: np.ndarray, points):
    slope, intercept = chromosome
    y_pred = slope * points.T[0] + intercept
    mse = np.mean((points.T[1] - y_pred) ** 2)

    return (mse,)


def linear_regression(points: np.ndarray) -> np.ndarray:
    toolbox = base.Toolbox()
    toolbox.register("individual", generate)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, points=points)

    pop = toolbox.population(500)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("max", np.max)

    algorithms.eaSimple(
        population=pop,
        toolbox=toolbox,
        cxpb=0.7,
        mutpb=0.3,
        ngen=100,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    return hof[0]
