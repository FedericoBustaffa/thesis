import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from explain.genetic import create_toolbox, update_toolbox
from ppga import algorithms, base, log


def make_predictions(model, data: pd.DataFrame, test_size: float = 0.3):
    features_index = [col for col in data.columns if col.startswith("feature_")]
    X = data[features_index].to_numpy()
    y = data["outcome"].to_numpy()

    # split train and test set
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    # train the model
    model.fit(X_train, y_train)

    # these will be the data to explain
    to_explain = np.asarray(model.predict(X_test))

    return np.asarray(X_test), to_explain


if __name__ == "__main__":
    logger = log.getUserLogger()
    logger.setLevel("INFO")

    df = pd.read_csv("datasets/classification_100_2_2_1_0.csv")
    # clf = MLPClassifier()
    classifiers = [RandomForestClassifier(), SVC(), MLPClassifier()]
    for clf in classifiers:
        X, y = make_predictions(clf, df, 0.1)
        toolbox = create_toolbox(X)
        toolbox = update_toolbox(toolbox, X[0], y[0], clf)

        hof = base.HallOfFame(500)
        start = time.perf_counter()
        algorithms.simple(toolbox, 1000, 0.1, 0.8, 0.2, 10, hof)
        end = time.perf_counter()
        stime = end - start
        logger.info(f"sequential time: {stime} seconds")

        hof = base.HallOfFame(500)
        start = time.perf_counter()
        algorithms.simple(toolbox, 1000, 0.1, 0.8, 0.2, 10, hof, -1)
        end = time.perf_counter()
        ptime = end - start
        logger.info(f"parallel time: {ptime} seconds")
        logger.info(f"speed up: {stime / ptime}")
