import argparse
from copy import deepcopy

import blackbox
import genetic
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from ppga import algorithms, base, log


def build_stats_df(results: list, blackbox) -> pd.DataFrame:
    stats = {
        "point": [],
        "class": [],
        "target": [],
        "min_fitness": [],
        "mean_fitness": [],
        "max_fitness": [],
        "accuracy": [],
    }

    for i, (outcome, one_pt_stats) in enumerate(results):
        hofs = one_pt_stats["hall_of_fame"]
        targets = one_pt_stats["target"]

        for hof, target in zip(hofs, targets):
            scores = np.asarray([ind.fitness for ind in hof])
            scores = scores[~np.isinf(scores)]
            synth_points = np.asarray([ind.chromosome for ind in hof])
            outcomes = blackbox.predict(synth_points)

            stats["point"].append(i)
            stats["class"].append(outcome)
            stats["target"].append(target)
            stats["min_fitness"].append(scores.min())
            stats["mean_fitness"].append(scores.mean())
            stats["max_fitness"].append(scores.max())
            stats["accuracy"].append(len(outcomes[outcomes == target]) / len(hof))

    return pd.DataFrame(stats)


def explain(X, y, blackbox):
    # collect all the possible outcomes
    outcomes = np.unique(y)

    # run genetic algorithm on every point
    toolbox = genetic.create_toolbox(X)

    results = []  # save results
    one_point_stats = {"hall_of_fame": [], "target": []}
    for point, outcome in zip(X, y):
        for target in outcomes:
            logger.debug(f"point: {point}")
            logger.debug(f"outcome: {outcome}")
            logger.debug(f"target: {target}")

            # update the point for the generation
            toolbox.set_generation(genetic.generate_copy, point=point)

            # update the evaluation function
            toolbox.set_evaluation(
                genetic.evaluate,
                point=point,
                target=target,
                blackbox=svm,
                epsilon=0.0,
                alpha=0.0,
            )

            # run the genetic algorithm on one point
            hof = base.HallOfFame(500)
            last, stats = algorithms.pelitist(
                toolbox=toolbox,
                population_size=2000,
                keep=0.1,
                cxpb=0.8,
                mutpb=0.2,
                max_generations=100,
                hall_of_fame=hof,
            )

            one_point_stats["hall_of_fame"].append(hof)
            one_point_stats["target"].append(target)

        results.append((outcome, deepcopy(one_point_stats)))
        one_point_stats["hall_of_fame"].clear()
        one_point_stats["target"].clear()

    return build_stats_df(results, svm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "filepath",
        type=str,
        help="filepath to the dataset",
    )

    parser.add_argument(
        "--log",
        type=str,
        choices=["debug", "benchmark", "info", "warning", "error"],
        default="info",
        help="set the debug level",
    )

    args = parser.parse_args()

    # logger from ppga
    logger = log.getUserLogger()
    logger.setLevel(args.log.upper())

    # call the blackbox run with the given dataset
    svm = SVC()
    predictions = blackbox.make_predictions(svm, args.filepath)

    X = predictions[[k for k in predictions if k != "outcome"]].to_numpy()
    y = predictions["outcome"].to_numpy()

    stats = explain(X, y, svm)
    print(stats)
