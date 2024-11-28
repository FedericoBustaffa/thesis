import json
import sys

import numpy as np
from data import make_data
from genetic import create_toolbox, evaluate, genetic_run
from sklearn.ensemble import RandomForestClassifier

from ppga import log


def explain(blackbox, X: np.ndarray, y: np.ndarray) -> dict[str, list]:
    # explainations dataframes
    explaination = {"class": [], "target": [], "hall_of_fame": []}

    # possible outcomes
    outcomes = np.unique(y)

    # run the genetic algorithm for every point
    for point, outcome in zip(X, y):
        toolbox = create_toolbox(X, point)
        for target in outcomes:
            toolbox.set_evaluation(evaluate, point, target, blackbox, 0.0, 0.0)
            hof, stats = genetic_run(toolbox, 200, 50)
            explaination["class"].append(outcome)
            explaination["target"].append(target)
            explaination["hall_of_fame"].append(hof)

    return explaination


def main(argv: list[str]):
    bb = RandomForestClassifier()

    X_train, X_test, y_train = make_data(n_samples=50, n_features=2, n_classes=2)
    bb.fit(X_train, y_train)
    y = np.asarray(bb.predict(X_test))

    logger = log.getUserLogger()
    logger.info(f"start explaining of {len(X_test)} points")
    explaination = explain(bb, X_test, y)

    filename = str(bb.__class__).split(" ")[1].split(".")[3].removesuffix("'>")
    with open(filename, "w") as file:
        json.dump(explaination, file, indent=2)


if __name__ == "__main__":
    main(sys.argv)
