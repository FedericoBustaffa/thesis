import warnings

import numpy as np
from numpy import linalg, random

from ppga import algorithms, base, tools

warnings.filterwarnings("ignore")


# generation by copy
def generate_copy(point: np.ndarray) -> np.ndarray:
    return point.copy()


# generation by normal distribution
def generate_normal(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return random.normal(mu, sigma, size=mu.shape)


# mutate
def mutate(
    individual: np.ndarray, mu: np.ndarray, sigma: np.ndarray, indpb: float = 0.5
):
    for i in range(len(individual)):
        if random.random() <= indpb:
            individual[i] = random.normal(mu[i], sigma[i])

    return individual


def evaluate(
    chromosome: np.ndarray,
    point: np.ndarray,
    target: np.ndarray,
    blackbox,
    epsilon: float = 0.0,
    alpha: float = 0.0,
):
    assert alpha >= 0.0 and alpha <= 1.0

    # classification
    synth_class = blackbox.predict(chromosome.reshape(1, -1))

    # compute euclidean distance
    distance = linalg.norm(chromosome - point, ord=2)

    # compute classification penalty
    right_target = 1.0 - alpha if target == synth_class[0] else alpha

    # check the epsilon distance
    if distance <= epsilon:
        return (np.inf,)

    return (distance / right_target,)


def create_toolbox(X: np.ndarray, point: np.ndarray) -> base.ToolBox:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)

    toolbox = base.ToolBox()
    toolbox.set_weights((-1.0,))
    toolbox.set_generation(generate_normal, mu=point, sigma=sigma * 0.15)
    toolbox.set_selection(tools.sel_tournament, tournsize=3)
    toolbox.set_crossover(tools.cx_one_point)
    toolbox.set_mutation(mutate, mu=mu, sigma=sigma, indpb=0.8)

    return toolbox


def genetic_run(
    toolbox: base.ToolBox, population_size: int, max_generations: int = 100
) -> tuple[base.HallOfFame, base.Statistics]:
    hof = base.HallOfFame(population_size)
    pop, stats = algorithms.pelitist(
        toolbox=toolbox,
        population_size=population_size,
        keep=0.1,
        cxpb=0.8,
        mutpb=0.2,
        max_generations=max_generations,
        hall_of_fame=hof,
    )

    return hof, stats
