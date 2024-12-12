import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg

from ppga import base, parallel
from ppga.algorithms import batch

warnings.filterwarnings("ignore")


# generation by copy
def generate_copy(point: np.ndarray) -> np.ndarray:
    return point.copy()


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


def plot_hof(
    X: np.ndarray,
    y: np.ndarray,
    hof: np.ndarray,
    point: np.ndarray | None = None,
    outcome: int | None = None,
):
    x0 = X.T[0][y == 0]
    x1 = X.T[0][y == 1]
    y0 = X.T[1][y == 0]
    y1 = X.T[1][y == 1]

    plt.clf()
    plt.scatter(x0, y0, c="b", ec="w")
    plt.scatter(x1, y1, c="r", ec="w")
    if point is not None:
        plt.scatter(
            point[0], point[1], c="b" if outcome == 0 else "r", ec="w", marker="X"
        )

    plt.scatter(hof.T[0], hof.T[1], c="y", ec="w")
    plt.pause(0.0001)


def run(
    X: np.ndarray,
    y: np.ndarray,
    point: np.ndarray,
    outcome,
    toolbox: base.ToolBox,
    population_size: int,
    max_generations: int,
):
    hof = base.HallOfFame(population_size)

    population = toolbox.generate(population_size)
    plt.figure(figsize=(16, 9))
    plt.title("Dataset")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    pool = parallel.Pool()
    for g in range(max_generations):
        print(g)
        selected = toolbox.select(population, population_size)
        couples = batch.mating(selected)
        offsprings = pool.map(batch.cx_mut_eval, couples, args=(toolbox, 0.8, 0.2))
        offsprings_copy = []
        for couple in offsprings:
            if couple != ():
                offsprings_copy.extend(couple)

        population = toolbox.replace(population, offsprings_copy)

        hof.update(population)

        best_points = np.asarray([i.chromosome for i in hof])
        # best_points = np.asarray([i.chromosome for i in population])
        plot_hof(X, y, best_points, point, outcome)

    plt.show()
    pool.join()
