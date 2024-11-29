import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def make_data(
    n_samples: int, n_features: int, n_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates classification training and test values"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        shuffle=True,
        random_state=0,
    )

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)

    return X_train, X_test, y_train


def generate_dataset(n_samples: int, n_features: int, n_classes: int) -> None:
    """Generate a dataset and store it in a CSV file"""
    X_train, X_test, y_train = make_data(n_samples, n_features, n_classes)
    training_set = {f"feature_{i+1}": X_train.T[i] for i in range(n_features)}
    training_set.update({"outcome": y_train})

    # create datasets folder if not present
    if "datasets" not in os.listdir("./explain/"):
        os.mkdir("./explain/datasets")

    # save trainig set
    df = pd.DataFrame(training_set)
    df.to_csv(
        f"./explain/datasets/classification_{len(X_train)}_{n_features}_{n_classes}_train.csv",
        header=True,
        index=False,
    )
    print(
        f"training set of {len(X_train)} samples, {n_features} features and {n_classes} classes generated"
    )

    # save test set
    test_set = {f"feature_{i+1}": X_test.T[i] for i in range(n_features)}
    df = pd.DataFrame(test_set)
    df.to_csv(
        f"./explain/datasets/classification_{len(X_test)}_{n_features}_{n_classes}_test.csv",
        header=True,
        index=False,
    )
    print(
        f"test set of {len(X_test)} samples, {n_features} features and {n_classes} classes generated"
    )
