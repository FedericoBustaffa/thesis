import numpy as np
import pandas as pd
from numpy import linalg, random

from ppga import algorithms, base, log, tools


def generate_gauss(mu, sigma, alpha: float) -> np.ndarray:
    return random.normal(mu, sigma * alpha, size=sigma.shape)


def evaluate(chromosome, point, target, classifier, epsilon: float, alpha: float = 0.5):
    assert alpha > 0.0 and alpha < 1.0

    # classification
    synth_class = classifier.predict(chromosome.reshape(1, -1))

    # calcolo della distanza con norma euclidea
    distance = linalg.norm(chromosome - point, ord=2)

    # compute classification penalty
    target_class = 1 - alpha if target == synth_class[0] else alpha

    # check the epsilon distance
    epsilon = 0.0 if target * distance > epsilon else epsilon

    return (target_class * distance + epsilon,)


def create_toolbox(point, sigma) -> base.ToolBox:
    toolbox = base.ToolBox()
    toolbox.set_weights((-1.0,))
    toolbox.set_generation(generate_gauss, mu=point, sigma=sigma, alpha=0.15)
    toolbox.set_selection(tools.sel_tournament, tournsize=3)
    toolbox.set_crossover(tools.cx_one_point)
    toolbox.set_mutation(tools.mut_gaussian, sigma=sigma, alpha=0.1, indpb=0.8)

    return toolbox


def genetic_run(
    toolbox: base.ToolBox, population_size: int, max_generations: int = 100
) -> tuple:
    hof = base.HallOfFame(population_size)

    pop, stats = algorithms.pelitist(
        toolbox=toolbox,
        population_size=population_size,
        keep=0.1,
        cxpb=0.7,
        mutpb=0.3,
        max_generations=max_generations,
        hall_of_fame=hof,
        log_level=log.INFO,
    )

    return pop, hof


def genetic_build(blackbox, point, target, sigma, alpha=0.5):
    toolbox = create_toolbox(point, sigma)
    epsilon = linalg.norm(sigma * 0.1, ord=2)
    toolbox.set_evaluation(evaluate, point, target, blackbox, epsilon, alpha)

    pop, hof = genetic_run(toolbox, 200, 50)

    pop_right = len(
        [i for i in pop if blackbox.predict(i.chromosome.reshape(1, -1)) == target]
    )

    hof_right = len(
        [i for i in hof if blackbox.predict(i.chromosome.reshape(1, -1)) == target]
    )

    point_class = blackbox.predict(point.reshape(1, -1))

    pop_df = pd.DataFrame(
        {
            "individuals": len(pop),
            "class": point_class,
            "target": target,
            "right_class": pop_right,
        }
    )

    hof_df = pd.DataFrame(
        {
            "individuals": len(hof),
            "class": point_class,
            "target": target,
            "right_class": hof_right,
        }
    )

    return pop_df, hof_df
