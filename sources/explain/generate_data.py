import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def make_data(n_samples: int, n_features: int, n_classes: int) -> tuple:
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

    return np.asarray(X), np.asarray(y)


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
        f"./explain/datasets/classification_{n_samples}_{n_features}_{n_classes}.csv",
        header=True,
        index=False,
    )
    print(
        f"dataset set of {len(X_train)} samples, {n_features} features and {n_classes} classes generated"
    )


def get_data(n_samples: int, n_features: int, n_classes: int) -> tuple:
    """get data if present, although generates a dataset and save it in a CSV file"""
    csv_files = [f for f in os.listdir("./explain/datasets/") if f.endswith(".csv")]
    print(csv_files)


if __name__ == "__main__":
    n_samples = int(sys.argv[1])
    n_features = int(sys.argv[2])
    n_classes = int(sys.argv[3])
    get_data(n_samples, n_features, n_classes)
