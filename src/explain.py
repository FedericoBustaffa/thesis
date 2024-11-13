import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data import make_data
from genetic_explain import genetic_build


def explain(blackbox, X, outcomes) -> list[pd.DataFrame]:
    # data to explain
    to_explain = blackbox.predict(X)

    # standard deviation of each feature
    sigma = X.std(axis=0)

    # explainations dataframes
    explain_dfs = []

    # run the genetic algorithm for every point
    for point, y in zip(X, to_explain):
        for target in outcomes:
            pop, hof = genetic_build(blackbox, point, target, sigma, 0.8)
            explain_dfs.extend([pop, hof])

    return explain_dfs


def main(argv: list[str]):
    X_train, X_test, y_train = make_data(n_samples=50, n_features=2, n_classes=2)

    bb = RandomForestClassifier()
    bb.fit(X_train, y_train)

    outcomes = np.unique(y_train)

    print(f"about to explain {len(X_test)} points")
    explainations = explain(bb, X_test, outcomes)
    print("genetic explain done")
    for ex in explainations:
        print(ex)


if __name__ == "__main__":
    main(sys.argv)
