import sys

import numpy as np
import pandas as pd
from data import make_data
from genetic_explain import genetic_build
from sklearn.ensemble import RandomForestClassifier

from ppga import log


def explain(blackbox, X, outcomes) -> pd.DataFrame:
    # data to explain
    to_explain = blackbox.predict(X)

    # standard deviation of each feature
    sigma = X.std(axis=0)

    # explainations dataframes
    explaination = {
        "individuals": [],
        "class": [],
        "target": [],
        "right": [],
    }

    # run the genetic algorithm for every point
    for point, y in zip(X, to_explain):
        for target in outcomes:
            hof = genetic_build(blackbox, point, target, sigma, 1.0)
            for k in explaination.keys():
                explaination[k].append(hof[k])

    return pd.DataFrame(explaination)


def main(argv: list[str]):
    X_train, X_test, y_train = make_data(n_samples=50, n_features=2, n_classes=2)

    bb = RandomForestClassifier()
    bb.fit(X_train, y_train)

    outcomes = np.unique(y_train)

    logger = log.getUserLogger()
    logger.info(f"start explaining of {len(X_test)} points")
    explaination = explain(bb, X_test, outcomes)
    explaination.to_csv("./results/explain.csv", header=True, index=False)
    logger.info(explaination)


if __name__ == "__main__":
    main(sys.argv)
