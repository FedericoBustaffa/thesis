import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def make_data(
    n_samples: int, n_features: int, n_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
