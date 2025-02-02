import genetic
import numpy as np

from ppga import base


def build_neighborhood(
    toolbox: base.ToolBox,
    population_size: int,
    point: np.ndarray,
    blackbox,
    target: int,
    workers_num: int,
) -> tuple[list, base.Statistics]:
    """
    Generates neighbors close to the given point and classified
    as the label given with the `target` parameter
    """
    # update the point for the generation
    toolbox = genetic.update_toolbox(toolbox, point, target, blackbox)
    hof, stats = genetic.run(toolbox, population_size, workers_num)
    return hof.to_list(), stats.to_dict()


def generate(
    X: np.ndarray,
    y: np.ndarray,
    model,
    population_size: int,
    workers_num: int = 0,
) -> dict[str, list]:
    """
    Generates synthetic neighbors for each point of the dataset.
    A neighborhood is generated for every possible outcome.
    """
    # collect all the possible outcomes
    outcomes = np.unique(y)

    # create a toolbox with fixed params
    toolbox = genetic.create_toolbox(X)

    # dataset of results
    results = []
    for point, label in zip(X, y):
        for target in outcomes:
            hof, stats = build_neighborhood(
                toolbox,
                population_size,
                point,
                model,
                target,
                workers_num,
            )

            results.append(
                {
                    "point": point.tolist(),
                    "class": int(label),
                    "target": int(target),
                    "model": str(model).removesuffix("()"),
                    "neighborhood": hof,
                    "stats": stats,
                }
            )

    return results
