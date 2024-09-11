import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def draw_graph(towns: pd.DataFrame, best: np.ndarray):
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


def fitness_trend(average: list[float], best: list[float]):
    generations = [g for g in range(len(average))]

    plt.figure(figsize=(12, 6))
    plt.title("Fitness trend")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    plt.plot(generations, average, label="Average fitness")
    plt.plot(generations, best, label="Best fitness")

    plt.grid()
    plt.legend()
    plt.show()


def biodiversity_trend(biodiversities: list[float]):
    generations = [g for g in range(len(biodiversities))]

    plt.figure(figsize=(12, 6))
    plt.title("Biodiversity trend")
    plt.xlabel("Generation")
    plt.ylabel("Biodiversity percentage")
    plt.plot(generations, biodiversities, label="Biodiversity", c="g")

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
