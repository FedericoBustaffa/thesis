import numpy as np

from ppga import algorithms, base, tools


def generate():
    return np.random.uniform(-5, 5, (2,))


def evaluate(chromosome: np.ndarray, points):
    slope, intercept = chromosome
    y_pred = slope * points.T[0] + intercept
    mse = np.mean((points.T[1] - y_pred) ** 2)

    return (mse,)


def linear_regression(points: np.ndarray) -> np.ndarray:
    toolbox = base.ToolBox()
    toolbox.set_weights((-1.0,))
    toolbox.set_generation(generate)
    toolbox.set_selection(tools.sel_tournament, tournsize=3)
    toolbox.set_crossover(tools.cx_blend, alpha=0.5)
    toolbox.set_mutation(tools.mut_normal, mu=[0, 0], sigma=[0.5, 0.5], indpb=0.3)
    toolbox.set_evaluation(evaluate, points=points)

    hof = base.HallOfFame(1)
    algorithms.simple(toolbox, 500, 0.2, 0.7, 0.3, 100, hof)

    return hof[0].chromosome
