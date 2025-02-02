import json

import generator

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    data = pd.read_csv("datasets/classification_10010_2_2_1_0.csv")

    X = np.array([data[k].to_numpy() for k in data if k != "outcome"]).T
    y = data["outcome"].to_numpy()

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=5, random_state=0)
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)

    for ps in [100, 1000, 10000]:
        neighbors = generator.generate(X_test, predictions, mlp, 100, -1)

        with open(f"results/neighborhood/res_{ps}.json", "w") as fp:
            json.dump(neighbors, fp, indent=2)

    blues = X_test[predictions == 0]
    reds = X_test[predictions == 1]

    point = blues[0]

    # plt.figure(figsize=(16, 9), dpi=300)
    # plt.title("Test Set")

    # plt.scatter(blues.T[0], blues.T[1], c="b", ec="w", s=50)
    # plt.scatter(point.T[0], point.T[1], c="b", ec="w", marker="X", s=50)
    # plt.scatter(reds.T[0], reds.T[1], c="r", ec="w", s=50)
    # plt.show()
