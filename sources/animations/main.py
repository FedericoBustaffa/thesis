import genetic
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from ppga import base, tools


def plot_dataset(
    X: np.ndarray,
    y: np.ndarray,
    point: np.ndarray | None = None,
    outcome: int | None = None,
):
    x0 = X.T[0][y == 0]
    x1 = X.T[0][y == 1]
    y0 = X.T[1][y == 0]
    y1 = X.T[1][y == 1]

    plt.figure(figsize=(16, 9))
    plt.title("Dataset")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.scatter(x0, y0, c="b", ec="w")
    plt.scatter(x1, y1, c="r", ec="w")
    if point is not None:
        plt.scatter(
            point[0], point[1], c="b" if outcome == 0 else "r", ec="w", marker="X"
        )
    plt.show()


def main():
    # create dataset
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        shuffle=True,
        random_state=0,
    )

    X = np.asarray(X)
    y = np.asarray(y)
    plot_dataset(X, y)

    # split
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    # train and test model
    classifier = MLPClassifier()
    classifier.fit(X_train, y_train)
    outcomes = np.asarray(classifier.predict(X_test))
    point = X_test[6]
    target = (outcomes[6] + 1) % 2
    plot_dataset(X_test, outcomes, point, outcomes[6])

    # run the genetic algorithm on one point
    toolbox = base.ToolBox()
    toolbox.set_weights(weights=(-1.0,))
    toolbox.set_generation(genetic.generate_copy, point=point)
    toolbox.set_selection(tools.sel_tournament, tournsize=3)
    toolbox.set_crossover(tools.cx_one_point)
    toolbox.set_mutation(
        tools.mut_normal,
        mu=X_test.mean(axis=0),
        sigma=X_test.std(axis=0),
        indpb=0.8,
    )
    toolbox.set_evaluation(
        genetic.evaluate,
        point=point,
        target=target,
        blackbox=classifier,
        epsilon=0.0,
        alpha=0.0,
    )
    toolbox.set_replacement(tools.elitist, keep=0.1)
    genetic.run(X_test, outcomes, point, outcomes[6], toolbox, 500, 100)


if __name__ == "__main__":
    main()
