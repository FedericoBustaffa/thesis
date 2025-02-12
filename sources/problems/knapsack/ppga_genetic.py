import logging

import pandas as pd
from common import Item, evaluate

from ppga import algorithms, base, log, tools, utility


def show_solution(solution, items):
    value = sum([i.value * s for i, s in zip(items, solution)])
    weight = sum([i.weight * s for i, s in zip(items, solution)])

    return value, weight


if __name__ == "__main__":
    logger = log.getUserLogger()
    logger.setLevel("INFO")
    log.setLevel("INFO")

    N = 1000
    G = 200

    nitems = [25, 50, 100, 200, 400]

    results = {"items": [], "capacity": [], "value": [], "weight": []}

    for n in nitems:
        df = pd.read_csv(f"problems/knapsack/datasets/items_{n}.csv")
        values = df["value"].to_list()
        weights = df["weight"].to_list()
        items = [Item(v, w) for v, w in zip(values, weights)]
        capacity = sum([i.weight for i in items]) * 0.5
        logger.info(f"capacity: {capacity:.3f}")

        toolbox = base.ToolBox()
        toolbox.set_weights(weights=(3.0, -1.0))
        toolbox.set_generation(tools.gen_repetition, (0, 1), len(items))
        toolbox.set_selection(tools.sel_ranking)
        toolbox.set_crossover(tools.cx_uniform)
        toolbox.set_mutation(tools.mut_bitflip)
        toolbox.set_evaluation(evaluate, items, capacity)

        hof = base.HallOfFame(1)

        best, stats = algorithms.simple(
            toolbox=toolbox,
            population_size=N,
            keep=0.1,
            cxpb=0.8,
            mutpb=0.2,
            max_generations=G,
            hall_of_fame=hof,
            workers_num=4,
        )

        value, weight = show_solution(hof[0].chromosome, items)
        logger.info(f"sequential best solution: ({value:.3f}, {weight:.3f})")
        logger.info(f"sequential best fitnes: {hof[0].fitness}")

        # plotting
        if logger.level < logging.INFO:
            utility.plot.fitness_trend(stats)
            utility.plot.biodiversity_trend(stats)

        results["items"].append(n)
        results["capacity"].append(capacity)
        results["value"].append(value)
        results["weight"].append(weight)

    df = pd.DataFrame(results)
    df.to_csv("problems/knapsack/results/ppga.csv", index=False, header=True)
    print(df)
