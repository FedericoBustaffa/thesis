import sys
import time
from functools import partial

import pandas as pd

from pipe_genetic import PipeGeneticAlgorithm
from tsp import *
from utils import plotting


def main(argv):
    if len(argv) != 6:
        print(
            f"USAGE: py {argv[0]} <town_file> <populations_size> <generations> <mutation_rate> <workers>"
        )
        exit(1)

    data = pd.read_csv(f"datasets/towns_{argv[1]}.csv")
    distances = compute_distances(data)

    # Initial population size
    N = int(argv[2])

    # Max generations
    G = int(argv[3])

    mutation_rate = float(argv[4])

    # number of workers
    W = int(argv[5])

    # partial functions to fix the arguments
    generate_func = partial(generate, len(distances))
    fitness_func = partial(fitness, distances)

    start = time.perf_counter()
    ga = PipeGeneticAlgorithm(
        N,
        len(data),
        generate_func,
        fitness_func,
        tournament,
        one_point_no_rep,
        rotation,
        mutation_rate,
        merge_replace,
        workers_num=W,
    )
    ga.run(G)
    # print(f"algorithm total time: {time.perf_counter() - start} seconds")

    # print(f"best score: {ga.best_score:.3f}")

    # drawing the graph
    # plotting.draw_graph(data, ga.best)

    # statistics data
    # plotting.fitness_trend(ga.average_fitness, ga.best_fitness)
    # plotting.biodiversity_trend(ga.biodiversity)

    # timing
    # plotting.timing(ga.timings)

    # for k in ga.timings.keys():
    #     print(f"{k}: {ga.timings[k]:.3f} seconds")
    # print(f"pure computation total time: {sum(ga.timings.values()):.3f} seconds")

    data = pd.read_csv("stats/tsp_stats.csv")
    stats = {
        "implementation": "pipe",
        "workers": W,
        "cities": [int(argv[1])],
        "population_size": [N],
        "generations": [G],
        "mutation_rate": [mutation_rate],
        "time": [ga.parallel_time],
    }

    data = pd.concat([data, pd.DataFrame.from_dict(stats)], ignore_index=True)
    data.to_csv("stats/tsp_stats.csv", index=False)


if __name__ == "__main__":
    main(sys.argv)
