import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import neighborhood_generator as ng
from ppga import log

if __name__ == "__main__":
    # set the debug log level of the core logger
    parser = argparse.ArgumentParser()

    # CLI arguments
    parser.add_argument(
        "dataset",
        type=str,
        help="select the dataset to run the simulation",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="info",
        help="set the log level of the core logger",
    )

    args = parser.parse_args()

    log.setLevel(args.log.upper())

    # build the dataset
    df = pd.read_csv(args.dataset)
    X = df[["feature_1", "feature_2"]].to_numpy()
    y = df["outcome"].to_numpy()

    # split train and test set
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=0)
    X_test = np.asarray(X_test)

    # train the model
    clf = MLPClassifier()
    clf.fit(X_train, y_train)

    # generate the genetic neighbors
    to_explain = np.asarray(clf.predict(X_test))
    toolbox = ng.create_toolbox(np.asarray(X_test))

    neighbors = ng.generate(clf, X_test, to_explain, 500, -1)
    print(pd.DataFrame(neighbors))
