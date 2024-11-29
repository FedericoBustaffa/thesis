import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate_dataset(n_samples: int, n_features: int, n_classes: int) -> None:
    """Generate a dataset and store it in a CSV file"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        shuffle=True,
    )
    X = np.asarray(X)
    y = np.asarray(y)

    training_set = {f"feature_{i + 1}": X.T[i] for i in range(n_features)}
    training_set.update({"outcome": y})

    # create datasets folder if not present
    if "datasets" not in os.listdir("./"):
        os.mkdir("./datasets")

    # save trainig set
    df = pd.DataFrame(training_set)
    df.to_csv(
        f"./datasets/classification_{n_samples}_{n_features}_{n_classes}.csv",
        header=True,
        index=False,
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"USAGE: python {sys.argv[0]} <n_samples> <n_features> <n_classes>")
        exit(1)

    n_samples = int(sys.argv[1])
    n_features = int(sys.argv[2])
    n_classes = int(sys.argv[3])

    generate_dataset(n_samples, n_features, n_classes)
    print(f"dataset set of {n_samples} samples,", end=" ")
    print(f"{n_features} features", end=" ")
    print(f"and {n_classes} classes generated")
