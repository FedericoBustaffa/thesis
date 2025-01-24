import statistics

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg

from ppga import algorithms, base, tools


def generate():
    return np.random.random(2)


def evaluate(chromosome: np.ndarray, points):
    slope, intercept = chromosome
    return (linalg.norm(points[:, 1] - (slope * points[:, 0] + intercept), ord=2),)


if __name__ == "__main__":
    x = np.linspace(0, 100, 50)
    y = x + np.random.normal(0, 50, (50,))

    m, q = statistics.linear_regression(x.tolist(), y.tolist())

    points = np.stack((x, y), axis=1)

    toolbox = base.ToolBox()
    toolbox.set_weights((-1.0,))
    toolbox.set_generation(generate)
    toolbox.set_selection(tools.sel_tournament, tournsize=3)
    toolbox.set_crossover(tools.cx_one_point)
    toolbox.set_mutation(
        tools.mut_normal, mu=points.mean(axis=0), sigma=points.std(axis=0)
    )
    toolbox.set_evaluation(evaluate, points=points)

    hof = base.HallOfFame(100)
    pop, stats = algorithms.simple(toolbox, 500, 0.1, 0.7, 0.3, 500, hof)

    genetic_slope, genetic_intercept = hof[0].chromosome
    plt.figure(figsize=(16, 9), dpi=150)
    plt.scatter(x, y, label="samples")
    plt.plot(x, m * x + q, c="r", label="linear regression")
    plt.plot(
        x, genetic_slope * x + genetic_intercept, c="g", label="genetic regression"
    )

    plt.grid()
    plt.legend()
    plt.show()
