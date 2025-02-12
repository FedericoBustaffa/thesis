import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm

matplotlib.rcParams.update({"font.size": 16})


def evaluate(chromosome, towns: np.ndarray) -> tuple[float]:
    distance = 0.0
    for i in range(len(chromosome) - 1):
        t1 = towns[chromosome[i]]
        t2 = towns[chromosome[i + 1]]
        distance += norm(t1 - t2, ord=2)

    return (distance,)


def draw_graph(towns: pd.DataFrame, best):
    x = [towns["x"][i] for i in best]
    y = [towns["y"][i] for i in best]

    plt.figure(figsize=(16, 9), dpi=300)
    # plt.title("Best path found")
    # plt.xlabel("X coordinates")
    # plt.ylabel("Y coordinates")

    plt.scatter(x, y, label="Towns")
    plt.plot(x, y, c="k", label="Path")

    plt.xticks([])
    plt.yticks([])

    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
