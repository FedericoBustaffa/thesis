import json
import os

import generator

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from ppga import log

if __name__ == "__main__":
    filepaths = os.listdir("datasets/")
    datasets = [pd.read_csv(f"datasets/{fp}") for fp in filepaths]

    logger = log.getUserLogger()
    logger.setLevel("INFO")

    results = []

    for i, data in enumerate(datasets):
        X = np.array([data[k].to_numpy() for k in data if k != "outcome"]).T
        y = data["outcome"].to_numpy()

        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=5, random_state=0
        )
        mlp = MLPClassifier()
        mlp.fit(X_train, y_train)
        predictions = mlp.predict(X_test)

        for ps in [1000, 2000, 4000]:
            logger.info(f"dataset {i + 1}/{len(datasets)}")
            logger.info(f"features: {len(X[0])}")
            logger.info(f"population size: {ps}")

            neighbors = generator.generate(X_test, predictions, mlp, ps, 4)
            results.append(neighbors)

    with open("results/neighborhood/res_MLPClassifier.json", "w") as fp:
        json.dump(results, fp, indent=2)
