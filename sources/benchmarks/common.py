import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        help="specify the model to benchmark",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="specify the suffix of the output file",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        help="specify the logging level",
    )

    return parser.parse_args()


def make_predictions(model, data: pd.DataFrame, test_size):
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
