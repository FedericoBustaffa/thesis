import multiprocessing as mp
import random

import pandas as pd
from common import Item, evaluate, show_solution

from deap import algorithms, base, creator, tools
from ppga import log

if __name__ == "__main__":
    logger = log.getUserLogger()
    logger.setLevel("INFO")
    log.setLevel("INFO")

    N = 1000
    G = 200

    nitems = [25, 50, 100, 200, 400]

    creator.create("FitnessMin", base.Fitness, weights=(2.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    results = {"items": [], "capacity": [], "value": [], "weight": []}

    for n in nitems:
        df = pd.read_csv(f"problems/knapsack/datasets/items_{n}.csv")
        values = df["value"].to_list()
        weights = df["weight"].to_list()
        items = [Item(v, w) for v, w in zip(values, weights)]
        capacity = sum([i.weight for i in items]) * 0.5
        logger.info(f"capacity: {capacity:.3f}")

        toolbox = base.Toolbox()
        toolbox.register("bit", random.choice, [0, 1])
        toolbox.register(
            "individual", tools.initRepeat, creator.Individual, toolbox.bit, n=n
        )

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
        toolbox.register("evaluate", evaluate, items=items, capacity=capacity)

        pool = mp.Pool(4)
        toolbox.register("map", pool.map)

        hof = tools.HallOfFame(1)

        best, logbook, _ = algorithms.eaSimple(
            population=toolbox.population(n=N),
            toolbox=toolbox,
            cxpb=0.8,
            mutpb=0.2,
            ngen=G,
            halloffame=hof,
        )

        pool.close()
        pool.join()

        value, weight = show_solution(hof[0], items)
        logger.info(f"sequential best solution: ({value:.3f}, {weight:.3f})")
        logger.info(f"sequential best fitnes: {hof[0].fitness}")

        results["items"].append(n)
        results["capacity"].append(capacity)
        results["value"].append(value)
        results["weight"].append(weight)

    df = pd.DataFrame(results)
    df.to_csv("problems/knapsack/results/deap.csv", index=False, header=True)
    print(df)
