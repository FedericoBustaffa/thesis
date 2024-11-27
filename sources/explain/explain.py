import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data import make_data
from genetic import create_toolbox, evaluate, genetic_run
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from ppga import log


def explain(blackbox, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    # explainations dataframes
    explaination = {"individuals": [], "class": [], "target": [], "right": []}

    # possible outcomes
    outcomes = np.unique(y)

    # run the genetic algorithm for every point
    for point, outcome in zip(X, y):
        toolbox = create_toolbox(X, point)
        for target in outcomes:
            toolbox.set_evaluation(evaluate, point, target, blackbox, 0.0, 0.0)
            hof, stats = genetic_run(toolbox, 200, 50)
            explaination["individuals"].append(hof.size)
            explaination["class"].append(outcome)
            explaination["target"].append(target)
            explaination["right"].append(
                len(
                    [
                        i
                        for i in hof
                        if blackbox.predict(i.chromosome.reshape(1, -1))[0] == target
                    ]
                )
            )

    return pd.DataFrame(explaination)


def main(argv: list[str]):
    bb = RandomForestClassifier()
    # bb = SVC()
    # bb = MLPClassifier()

    X_train, X_test, y_train = make_data(n_samples=50, n_features=2, n_classes=2)
    bb.fit(X_train, y_train)
    y = np.asarray(bb.predict(X_test))

    plt.figure(figsize=(16, 9))
    plt.scatter(X_test.T[0], X_test.T[1], c=y)
    plt.show()

    logger = log.getUserLogger()
    logger.info(f"start explaining of {len(X_test)} points")
    explaination = explain(bb, X_test, y)
    explaination.to_csv("./results/explain.csv", header=True, index=False)


if __name__ == "__main__":
    main(sys.argv)
