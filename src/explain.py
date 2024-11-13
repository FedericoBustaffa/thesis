import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data import make_data
from genetic_explain import genetic_explain_diff, genetic_explain_same


def explain(blackbox, X, outcomes) -> list[pd.DataFrame]:
    # data to explain
    to_explain = blackbox.predict(X)

    # standard deviation of each feature
    sigma = X.std(axis=0)

    # explainations dataframes
    explain_dfs = []

    # run the genetic algorithm for every point
    for point, y in zip(X, to_explain):
        explaination_df = genetic_explain_same(blackbox, point, sigma, 0.6)
        explain_dfs.append(explaination_df)

        for target in outcomes:
            if target != y:
                pop, hof = genetic_explain_diff(blackbox, point, target, sigma, 0.8)

    return explain_dfs


def main(argv: list[str]):
    X_train, X_test, y_train = make_data(n_samples=500, n_features=2, n_classes=3)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    outcomes = np.unique(y_train)
    explainations = explain(rf, X_test, outcomes)
    for ex in explainations:
        print(ex)


if __name__ == "__main__":
    main(sys.argv)
