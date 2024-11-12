import sys

import numpy as np
from numpy import linalg, random
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ppga import algorithms, base, log, tools


def make_data(n_samples: int, n_features: int, n_classes: int):
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

    return train_test_split(X, y, test_size=0.2, random_state=0)


def generate_gauss(mu, sigma, alpha: float) -> np.ndarray:
    return random.normal(mu, sigma * alpha, size=sigma.shape)


def same_evaluate(chromosome, point, classifier, epsilon: float, alpha: float = 0.5):
    assert alpha > 0.0 and alpha < 1.0

    # classification
    pt_class = classifier.predict(point.reshape(1, -1))
    synth_class = classifier.predict(chromosome.reshape(1, -1))

    # calcolo della distanza con norma euclidea
    distance = linalg.norm(chromosome - point, ord=2)

    # compute classification weight
    same_class = 1 - alpha if pt_class == synth_class[0] else alpha

    # check the epsilon distance
    epsilon = 0.0 if same_class * distance > epsilon else epsilon

    return (same_class * distance + epsilon,)


def other_evaluate(
    chromosome, point, target, classifier, epsilon: float, alpha: float = 0.5
):
    assert alpha >= 0.0 and alpha < 1.0

    # classification
    synth_class = classifier.predict(chromosome.reshape(1, -1))

    # calcolo della distanza con norma euclidea
    distance = linalg.norm(chromosome - point, ord=2)

    # compute classification penalty
    target_class = 1 - alpha if target == synth_class[0] else alpha

    # check the epsilon distance
    epsilon = 0.0 if target * distance > epsilon else epsilon

    return (target_class * distance + epsilon,)


def genetic_explain_same(blackbox, point, y, sigma, alpha=0.5):
    toolbox = base.ToolBox()
    toolbox.set_weights((-1.0,))
    toolbox.set_generation(generate_gauss, mu=point, sigma=sigma, alpha=0.15)
    toolbox.set_selection(tools.sel_tournament, tournsize=3)
    toolbox.set_crossover(tools.cx_one_point)
    toolbox.set_mutation(tools.mut_gaussian, sigma=sigma, alpha=0.1, indpb=0.8)

    epsilon = linalg.norm(sigma * 0.1, ord=2)
    toolbox.set_evaluation(same_evaluate, point, blackbox, epsilon, alpha)

    hof = base.HallOfFame(500)

    pop, stats = algorithms.pelitist(
        toolbox=toolbox,
        population_size=1000,
        keep=0.1,
        cxpb=0.7,
        mutpb=0.3,
        max_generations=50,
        hall_of_fame=hof,
        log_level=log.INFO,
    )

    return pop, hof


def explain(blackbox, X, outcomes):
    # data to explain
    y_predicted = blackbox.predict(X)

    # standard deviation of each feature
    sigma = X.std(axis=0)

    same_populations = []
    same_hall_of_fames = []

    # run the genetic algorithm for every point
    for point, y in zip(X, y_predicted):
        pop, hof = genetic_explain_same(blackbox, point, y, sigma, 0.6)
        same_populations.append(pop)
        same_hall_of_fames.append(hof)


def main(argv: list[str]):
    X_train, X_test, y_train, _ = make_data(200, 2, 2)
    outcomes = np.unique(y_train)

    blackbox = RandomForestClassifier()
    blackbox.fit(X_train, y_train)

    explain(blackbox, X_test, outcomes)


if __name__ == "__main__":
    main(sys.argv)
