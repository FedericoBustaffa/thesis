import sys
import time
from functools import partial

import numpy as np
import pandas as pd

import tsp
from genetic import GeneticAlgorithm
from utils import plotting


def main(argv):
    if len(argv) < 5:
        print(f"USAGE: py {argv[0]} <T> <N> <G> <M>")
        exit(1)

    data = pd.read_csv(f"datasets/towns_{argv[1]}.csv")
    towns = np.array([[data["x"].iloc[i], data["y"].iloc[i]] for i in range(len(data))])

    # Initial population size
    N = int(argv[2])

    # Max generations
    G = int(argv[3])

    mutation_rate = float(argv[4])

    # partial functions to fix the arguments
    generate_func = partial(tsp.generate, len(towns))
    fitness_func = partial(tsp.fitness, towns)

    start = time.perf_counter()
    ga = GeneticAlgorithm(
        N,
        len(data),
        generate_func,
        fitness_func,
        tsp.tournament,
        tsp.one_point_no_rep,
        tsp.rotation,
        mutation_rate,
        tsp.merge_replace,
    )
    ga.run(G)
    print(f"algorithm total time: {time.perf_counter() - start} seconds")

    print(f"best score: {ga.best_score:.3f}")

    # drawing the graph
    plotting.draw_graph(data, ga.best)

    # statistics data
    plotting.fitness_trend(ga.average_fitness, ga.best_fitness)
    plotting.biodiversity_trend(ga.biodiversity)

    # timing
    plotting.timing(ga.timings)

    for k in ga.timings.keys():
        print(f"{k}: {ga.timings[k]:.3f} seconds")
    print(f"total time: {sum(ga.timings.values()):.3f} seconds")

    # data = pd.read_csv("stats/tsp_stats.csv")
    # stats = {
    #     "implementation": "sequential",
    #     "workers": [1],
    #     "cities": [int(argv[1])],
    #     "population_size": [N],
    #     "generations": [G],
    #     "mutation_rate": [mutation_rate],
    #     "time": [
    #         sum(
    #             [
    #                 ga.timings["crossover"],
    #                 ga.timings["mutation"],
    #                 ga.timings["evaluation"],
    #             ]
    #         )
    #     ],
    # }
    #
    # if data.empty:
    #     data = pd.DataFrame.from_dict(stats)
    # else:
    #     data = pd.concat([data, pd.DataFrame.from_dict(stats)], ignore_index=True)
    #
    # data.to_csv("stats/tsp_stats.csv", index=False)


if __name__ == "__main__":
    main(sys.argv)
