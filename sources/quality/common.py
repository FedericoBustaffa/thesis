import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_predictions(
    model, data: pd.DataFrame, test_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes in a ML model and a dataset, trains the model and returns the test
    set and the predictions.

    Args:
        model: the ML model used for classification
        data: the dataset
        test_size: the size of the test set
        train_size: the size of the training set

    Returns:
        A tuple containing the test set and the predictions
    """
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


def get_args():
    # parse CLI argument
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        choices=["RandomForestClassifier", "SVC", "MLPClassifier"],
        help="specify the model to explain",
    )

    parser.add_argument(
        "workers",
        type=int,
        help="specify the number of workers to use",
    )

    parser.add_argument(
        "output",
        default="output",
        help="specify the name of the output file without extension",
    )

    parser.add_argument(
        "--log",
        default="info",
        help="set the log level of the core logger",
    )

    return parser.parse_args()
