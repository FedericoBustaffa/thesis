import argparse
import json

import generator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from ppga import log

if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath", type=str, help="specify the filepath to the dataset"
    )
    parser.add_argument(
        "pop_size",
        type=int,
        help="specify the number of synthetic individuals to generate",
    )
    args = parser.parse_args()

    # logger
    logger = log.getUserLogger()
    logger.setLevel("INFO")

    # read dataset
    data = pd.read_csv(args.filepath)

    # generate data for genetic alg
    X = np.array([data[k].to_numpy() for k in data if k != "outcome"]).T
    y = data["outcome"].to_numpy()
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=50, random_state=0)
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)

    logger.info(f"features: {len(X[0])}")
    logger.info(f"population size: {args.pop_size}")

    # generate the synthetic neighbors
    neighbors = generator.generate(X_test, predictions, mlp, args.pop_size)

    with open("./synthetic.json", "w") as fp:
        json.dump(neighbors, fp, indent=2)
