import sys
from functools import partial
import time

import pandas as pd
from shared_genetic import SharedMemoryGeneticAlgorithm
from tsp import *
from utils import plotting

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(
            f"USAGE: py {sys.argv[0]} <town_file> <populations_size> <generations> <mutation_rate> <workers>"
        )
        exit(1)

    data = pd.read_csv(f"datasets/towns_{sys.argv[1]}.csv")
    distances = compute_distances(data)

    # Initial population size
    N = int(sys.argv[2])

    # Max generations
    G = int(sys.argv[3])

    mutation_rate = float(sys.argv[4])

    # number of workers
    W = int(sys.argv[5])

    # partial functions to fix the arguments
    generate_func = partial(generate, len(distances))
    fitness_func = partial(fitness, distances)

    start = time.perf_counter()
    ga = SharedMemoryGeneticAlgorithm(
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
