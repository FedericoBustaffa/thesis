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
    classifiers = [RandomForestClassifier(), SVC(), MLPClassifier()]
    population_sizes = [1000, 2000, 4000, 8000, 16000]
    workers = [1, 2, 4, 8, 16, 32]

    results = {
        "classifier": [],
        "population_size": [],
        "workers": [],
        "time": [],
        "time_std": [],
    }

    for clf in classifiers:
        X, y = make_predictions(clf, df, 0.3)
        toolbox = create_toolbox(X)
        toolbox = update_toolbox(toolbox, X[0], y[0], clf)
        for ps in population_sizes:
            for w in workers:
                times = []
                for i in range(10):
                    hof = base.HallOfFame(ps)
                    start = time.perf_counter()
                    algorithms.simple(toolbox, ps, 0.1, 0.8, 0.2, 5, hof, w)
                    end = time.perf_counter()
                    times.append(end - start)

                results["classifier"].append(str(clf).removesuffix("()"))
                results["population_size"].append(ps)
                results["workers"].append(w)
                results["time"].append(np.mean(times))
                results["time_std"].append(np.std(times))
                logger.info(f"classifier: {str(clf).removesuffix('()')}")
                logger.info(f"population_size: {ps}")
                logger.info(f"workers: {w}")

    results = pd.DataFrame(results)
    results.to_csv("datasets/ppga_benchmark.csv", index=False, header=True)
    print(results)
