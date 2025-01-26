import random

import numpy as np
import pandas as pd
from common import Town, evaluate

# from scoop import futures
from utils import plotting

from deap import algorithms, base, creator, tools


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
    G = 500

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    towns_num = [10, 20, 50, 100]
    population_sizes = [100, 200, 400, 800]
    df = {"towns": [], "population_size": [], "distance": []}

    for tn in towns_num:
        for ps in population_sizes:
            data = pd.read_csv(f"problems/tsp/datasets/towns_{tn}.csv")
            x_coords = data["x"]
            y_coords = data["y"]
            towns = [Town(x, y) for x, y in zip(x_coords, y_coords)]

            toolbox = base.Toolbox()
            toolbox.register("indices", random.sample, range(tn), tn)
            toolbox.register(
                "individual", tools.initIterate, creator.Individual, toolbox.indices
            )

            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("mate", tools.cxOrdered)
            toolbox.register("mutate", mut_rotation)
            toolbox.register("select", tools.selTournament, tournsize=2)
            toolbox.register("evaluate", evaluate, towns=towns)

            hall_of_fame = tools.HallOfFame(5)

            algorithms.eaSimple(
                toolbox.population(n=ps),
                toolbox=toolbox,
                cxpb=0.7,
                mutpb=0.3,
                ngen=G,
                halloffame=hall_of_fame,
            )

            df["towns"].append(tn)
            df["population_size"].append(ps)
            df["distance"].append(evaluate(hall_of_fame[0], towns)[0])

    df = pd.DataFrame(df)
    df.to_csv("problems/tsp/results/deap_tsp.csv", index=False, header=True)
    print(df)

    # plotting the best solution ever recorded
    # plotting.draw_graph(data, hall_of_fame[0])
