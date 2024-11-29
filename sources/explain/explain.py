import json
import sys

import numpy as np
from generate_data import get_data
from genetic import create_toolbox, evaluate, genetic_run
from sklearn.svm import SVC

from ppga import log


def explain(blackbox, X: np.ndarray, y: np.ndarray) -> list:
    explainations = []

    # possible outcomes
    outcomes = np.unique(y)

    # run the genetic algorithm for every point
    for point, outcome in zip(X, y):
        toolbox = create_toolbox(X, point)
        for target in outcomes:
            toolbox.set_evaluation(evaluate, point, target, blackbox, 0.0, 0.0)
            hof, stats = genetic_run(toolbox, 100, 50)
            explaination = dict()
            explaination.update({"point": point.tolist()})
            explaination.update({"class": int(outcome)})
            explaination.update({"target": int(target)})
            explaination.update({"hall_of_fame": [i.to_dict() for i in hof.hof]})

            explainations.append(explaination)

    return explainations


def main(argv: list[str]):
    logger = log.getUserLogger()
    logger.setLevel("DEBUG")

    if len(argv) != 4:
        logger.error(f"USAGE: python {argv[0]} <samples> <features> <classes>")
        exit(1)

    n_samples = int(argv[1])
    n_features = int(argv[2])
    n_classes = int(argv[3])

    # get data if present, although generates a dataset and save it in a CSV file
    X_train, X_test, y_train = get_data(n_samples, n_features, n_classes)
    logger.info(f"start explaining of {len(X_test)} points")
    bb = SVC()
    bb.fit(X_train, y_train)
    y = np.asarray(bb.predict(X_test))

    explaination = explain(bb, X_test, y)
    filename = str(bb.__class__).split(" ")[1].split(".")[3].removesuffix("'>")

    to_json = explaination
    with open(f"results/{filename}.json", "w") as file:
        json.dump(to_json, fp=file, indent=2)


if __name__ == "__main__":
    main(sys.argv)
