import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier

from ppga import algorithms, base, log, tools


def evaluate():
    pass


def main():
    df = load_iris(return_X_y=True, as_frame=True)
    X = df[0]
    y = df[1]
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    x_train = X[25:]
    y_train = y[25:]

    x_test = X[:25]
    y_test = np.array(y[:25])

    print(x_train)
    print(y_train)

    classes = y.unique()
    print(classes)

    classifier = MLPClassifier()
    classifier.fit(X.to_numpy(), y)
    y = np.array(classifier.predict(x_test))

    print(y)
    print(y_test)


if __name__ == "__main__":
    main()
