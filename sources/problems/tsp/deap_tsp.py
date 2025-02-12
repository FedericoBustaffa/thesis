import multiprocessing as mp
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import draw_graph, evaluate

from deap import algorithms, base, creator, tools
from ppga import utility


def fitness_trend(stats):
    generations = [g for g in range(len(stats["max"]))]

    plt.figure(figsize=(16, 10), dpi=200)
    plt.title("Fitness trend")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    plt.plot(generations, stats["max"], label="Best fitness", c="g")
    plt.plot(generations, stats["mean"], label="Mean fitness", c="b")
    plt.plot(generations, stats["min"], label="Worst fitness", c="r")

    plt.grid()
    plt.legend()
    plt.show()


def cx_one_point_ordered(father, mother) -> tuple:
    cx_point = random.randint(1, len(father) - 1)

    offspring1 = father[:cx_point]
    offspring2 = father[cx_point:]

    for gene in mother:
        if gene not in offspring1:
            offspring1.append(gene)
        else:
            offspring2.append(gene)

    father[:] = offspring1[:]
    mother[:] = offspring2[:]

    return father, mother


def mut_rotation(chromosome: np.ndarray):
    a, b = np.random.choice(
        [i for i in range(len(chromosome) + 1)], size=2, replace=False
    )
    if a > b:
        a, b = b, a
    chromosome[a:b] = np.flip(chromosome[a:b])

    return (chromosome,)


if __name__ == "__main__":
    # Max generations
    G = 200

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    ntowns = [25, 50, 100, 200, 400]
    population_sizes = [2000]
    df = {"towns": [], "distance": []}

    for tn in ntowns:
        for ps in population_sizes:
            data = pd.read_csv(f"problems/tsp/datasets/towns_{tn}.csv")
            x_coords = data["x"]
            y_coords = data["y"]
            towns = np.array([[x, y] for x, y in zip(x_coords, y_coords)])

            toolbox = base.Toolbox()
            toolbox.register("indices", random.sample, range(tn), tn)
            toolbox.register(
                "individual", tools.initIterate, creator.Individual, toolbox.indices
            )

            pool = mp.Pool()
            toolbox.register("map", pool.map)

            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("select", tools.selTournament, tournsize=2)
            toolbox.register("mate", cx_one_point_ordered)
            toolbox.register("mutate", mut_rotation)
            toolbox.register("evaluate", evaluate, towns=towns)

            stats = tools.Statistics(key=lambda ind: ind.fitness.wvalues)
            stats.register("min", np.min)
            stats.register("mean", np.mean)
            stats.register("max", np.max)

            hall_of_fame = tools.HallOfFame(1)

            pop, logbook, _ = algorithms.eaSimple(
                toolbox.population(n=ps),
                toolbox=toolbox,
                cxpb=0.7,
                mutpb=0.3,
                ngen=G,
                stats=stats,
                halloffame=hall_of_fame,
            )

            stats = {
                "nevals": logbook.select("nevals"),
                "min": logbook.select("min"),
                "mean": logbook.select("mean"),
                "max": logbook.select("max"),
            }

            df["towns"].append(tn)
            df["distance"].append(evaluate(hall_of_fame[0], towns)[0])

            pool.close()
            pool.join()

            # plotting the best solution ever recorded
            # draw_graph(data, hall_of_fame[0])
            # fitness_trend(stats)
            # utility.plot.evals(stats["nevals"])

    df = pd.DataFrame(df)
    df.to_csv("problems/tsp/results/deap_tsp.csv", index=False, header=True)
    print(df)
