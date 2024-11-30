import json
import sys

import numpy as np
import pandas as pd
from genetic import create_toolbox, evaluate, genetic_run
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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
    # logger from ppga
    logger = log.getUserLogger()
    logger.setLevel("INFO")

    # command line args
    if len(argv) != 2:
        logger.error(f"USAGE: python {argv[0]} <filename>")
        exit(1)

    # read dataset file
    data = pd.read_csv(argv[1])
    X = np.array([np.asarray(data[k]) for k in data if k.startswith("feature")]).T
    y = data["outcome"].to_numpy()

    # splitting training and test sets
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)

    # start the explaination
    logger.info(f"start explaining of {len(X_test)} points")
    blackboxes = [RandomForestClassifier(), SVC(), MLPClassifier()]

    for bb in blackboxes:
        bb.fit(X_train, y_train)
        y = np.asarray(bb.predict(X_test))

        # parse the name of the blackbox
        bb_name = str(bb.__class__).split(" ")[1].split(".")[3].removesuffix("'>")
        logger.info(f"{bb_name} start")

        # explaination build and save
        explaination = explain(bb, np.asarray(X_test), y)
        logger.info(f"{bb_name} done")

        # json file
        with open(f"results/{bb_name}.json", "w") as file:
            json.dump(explaination, fp=file, indent=2)


if __name__ == "__main__":
    main(sys.argv)
