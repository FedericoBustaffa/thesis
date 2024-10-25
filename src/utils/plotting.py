import matplotlib.pyplot as plt
import pandas as pd

from ppga.base.statistics import Statistics


def draw_graph(towns: pd.DataFrame, best: list[int]):
    x = [towns["x"][i] for i in best]
    y = [towns["y"][i] for i in best]

    plt.figure(figsize=(12, 6))
    plt.title("Best path found")
    plt.xlabel("X coordinates")
    plt.ylabel("Y coordinates")

    plt.scatter(x, y, label="Towns")
    plt.plot(x, y, c="k", label="Path")

    plt.legend()
    plt.show()


def fitness_trend(stats: Statistics):
    best = stats.best
    mean = stats.mean
    worst = stats.worst

    generations = [g for g in range(len(best))]

    plt.figure(figsize=(12, 6))
    plt.title("Fitness trend")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    plt.plot(generations, best, label="Best fitness", c="g")
    plt.plot(generations, mean, label="Mean fitness", c="b")
    plt.plot(generations, worst, label="Worst fitness", c="r")

    plt.grid()
    plt.legend()
    plt.show()


def biodiversity_trend(stats: Statistics):
    diversity = stats.diversity
    generations = [g for g in range(len(diversity))]

    plt.figure(figsize=(12, 6))
    plt.title("Biodiversity trend")
    plt.xlabel("Generation")
    plt.ylabel("Biodiversity percentage")
    plt.plot(generations, diversity, label="Biodiversity", c="g")

    plt.grid()
    plt.legend()
    plt.show()


def timing(timings: dict[str, float]):
    plt.figure(figsize=(12, 6))
    plt.title("Timing")
    plt.pie(
        [v for v in timings.values()],
        labels=[k for k in timings.keys()],
        autopct="%1.1f%%",
    )
    plt.show()
