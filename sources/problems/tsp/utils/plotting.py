import matplotlib.pyplot as plt
import pandas as pd

from ppga.base.statistics import Statistics

figsize = (16, 8)


def draw_graph(towns: pd.DataFrame, best):
    x = [towns["x"][i] for i in best]
    y = [towns["y"][i] for i in best]

    plt.figure(figsize=figsize)
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

    plt.figure(figsize=figsize)
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

    plt.figure(figsize=figsize)
    plt.title("Biodiversity trend")
    plt.xlabel("Generation")
    plt.ylabel("Biodiversity percentage")
    plt.plot(generations, diversity, label="Biodiversity", c="g")

    plt.grid()
    plt.legend()
    plt.show()


def timing(timings: dict[str, float]):
    plt.figure(figsize=figsize)
    plt.title("Timing")
    plt.pie(
        [v for v in timings.values()],
        labels=[k for k in timings.keys()],
        autopct="%1.1f%%",
    )
    plt.show()


def evals(evals: list[int]):
    plt.figure(figsize=figsize)
    plt.title("Evaluations")
    plt.xlabel("Generation")
    plt.ylabel("Evals")

    plt.plot([g for g in range(len(evals))], evals, label="evals")

    plt.legend()
    plt.grid()
    plt.show()


def multievals(evals: list[list[int]]):
    plt.figure(figsize=figsize)
    plt.title("Evaluations per worker")
    plt.xlabel("Worker")
    plt.ylabel("Number of evaluations")
    plt.xticks(list(range(len(evals[0]))))

    evals_per_worker = [0 for _ in range(len(evals[0]))]
    for e in evals:
        for i, n in enumerate(e):
            evals_per_worker[i] += n

    plt.bar(list(range(len(evals_per_worker))), evals_per_worker, label="bar")

    plt.legend()
    plt.show()
