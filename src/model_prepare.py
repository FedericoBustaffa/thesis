from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from ppga import log


def make_data(n_samples):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        shuffle=True,
        random_state=0,
    )

    train_size = int(n_samples * 80 / 100)
    test_size = n_samples - train_size

    logger = log.getLogger()
    logger.setLevel(log.INFO)
    logger.info(f"train size generated: {train_size}")
    logger.info(f"test size generated: {test_size}")

    return train_test_split(X, y, test_size=test_size, train_size=train_size)


def get_mlp(X, y) -> MLPClassifier:
    classifier = MLPClassifier(max_iter=2000)
    classifier.fit(X, y)

    return classifier


def get_random_forest(X, y) -> RandomForestClassifier:
    classifier = RandomForestClassifier()
    classifier.fit(X, y)

    return classifier
