import argparse

import blackbox
import genetic
import numpy as np
from sklearn.svm import SVC

from ppga import algorithms, base, log


def save_results(hof: base.HallOfFame):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log",
        type=str,
        choices=["debug", "benchmark", "info", "warning", "error"],
        default="info",
        help="set the debug level",
    )

    parser.add_argument(
        "filepath",
        type=str,
        help="filepath to the dataset",
    )

    args = parser.parse_args()

    # logger from ppga
    logger = log.getUserLogger()
    logger.setLevel(args.log.upper())

    # call the blackbox run with the given dataset
    svm = SVC()
    predictions = blackbox.make_predictions(svm, args.filepath)
    logger.debug("predictions done")

    X = predictions[[k for k in predictions if k != "outcome"]].to_numpy()
    y = predictions["outcome"].to_numpy()

    # collect all the possible outcomes
    outcomes = np.unique(y)

    # run genetic algorithm on every point
    toolbox = genetic.create_toolbox(X)
    population_size = len(y)  # This should not be fixed
    for point, outcome in zip(X, y):
        for target in outcomes:
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

            # run the genetic algorithm
            hof = base.HallOfFame(population_size)
            last, stats = algorithms.pelitist(
                toolbox=toolbox,
                population_size=population_size,
                keep=0.1,
                cxpb=0.8,
                mutpb=0.2,
                max_generations=100,
                hall_of_fame=hof,
            )

            save_results(hof, point, target, svm)
