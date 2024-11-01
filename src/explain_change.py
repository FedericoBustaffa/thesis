import matplotlib.pyplot as plt
import numpy as np

from model_prepare import get_mlp, make_data
from ppga import algorithms, base, log, tools


def main():
    X_train, X_test, y_train, y_test = make_data(100)
    mlp = get_mlp(X_train, y_train)
    y_predict = np.array(mlp.predict(X_test))

    toolbox = base.ToolBox()
    toolbox.set_generation(generate)


if __name__ == "__main__":
    main()
