import math
import time

import numpy as np

from genetic_solver import GeneticSolver
from modules import Worker


class PipeGeneticSolver(GeneticSolver):
    def __init__(
        self,
        population_size: int,
        generation_func,
        fitness_func,
        selection_func,
        mating_func,
        crossover_func,
        crossover_rate: float,
        mutation_func,
        mutation_rate: float,
        replace_func,
        workers_num: int,
    ) -> None:
        super().__init__(
            population_size,
            generation_func,
            fitness_func,
            selection_func,
            mating_func,
            crossover_func,
            crossover_rate,
            mutation_func,
            mutation_rate,
            replace_func,
        )

        self.__workers = [
            Worker(self._crossoverator, self._mutator, self._evaluator)
            for _ in range(workers_num)
        ]
        for w in self.__workers:
            w.start()

    def run(self, max_generations: int):
        self._population = self._generator.perform()
        self._scores = self._evaluator.perform(self._population)

        timing = 0.0
        for g in range(max_generations):
            logger.trace(f"generation: {g + 1}")
            chosen = self._selector.perform(self._population, self._scores)
            couples = self._mater.perform(chosen)

            # parallel work
            start = time.perf_counter()
            chunksize = math.ceil(len(couples) / len(self.__workers))
            for i in range(len(self.__workers)):
                self.__workers[i].send(
                    couples[i * chunksize : i * chunksize + chunksize]
                )

            offsprings = []
            offsprings_scores = []
            for w in self.__workers:
                offsprings_chunk, offsprings_scores_chunk = w.recv()
                offsprings.extend(offsprings_chunk)
                offsprings_scores.extend(offsprings_scores_chunk)
            timing += time.perf_counter() - start

            self._population, self._scores = self._replacer.perform(
                self._population, self._scores, offsprings, offsprings_scores
            )

        for w in self.__workers:
            w.send(None)
            w.join()

        logger.info(f"parallel time: {timing}")


if __name__ == "__main__":
    import sys
    import time
    from functools import partial

    import numpy as np
    import pandas as pd
    from loguru import logger

    import tsp
    from utils import plotting

    if len(sys.argv) < 8:
        logger.error(f"USAGE: py {sys.argv[0]} <T> <N> <G> <C> <M> <W> <log_level>")
        exit(1)

    logger.remove()
    logger.add(sys.stderr, level=sys.argv[7].upper())

    data = pd.read_csv(f"datasets/towns_{sys.argv[1]}.csv")
    towns = np.array([[data["x"].iloc[i], data["y"].iloc[i]] for i in range(len(data))])

    # Initial population size
    N = int(sys.argv[2])

    # Max generations
    G = int(sys.argv[3])

    # crossover rate
    CR = float(sys.argv[4])

    # mutation rate
    MR = float(sys.argv[5])

    # number of workers
    W = int(sys.argv[6])

    # partial functions to fix the arguments
    generate_func = partial(tsp.generate, len(towns))
    fitness_func = partial(tsp.fitness, towns)

    ga = PipeGeneticSolver(
        population_size=N,
        generation_func=generate_func,
        fitness_func=fitness_func,
        selection_func=tsp.tournament,
        mating_func=tsp.couples_mating,
        crossover_func=tsp.one_point_no_rep,
        crossover_rate=CR,
        mutation_func=tsp.rotation,
        mutation_rate=MR,
        replace_func=tsp.merge_replace,
        workers_num=W,
    )
    start = time.perf_counter()
    ga.run(G)
    logger.info(f"algorithm total time: {time.perf_counter() - start:.6f} seconds")

    best, best_score = ga.get()
    logger.info(f"best score: {best_score:.6f}")

    # drawing the graph
    plotting.draw_graph(data, best)

    # statistics data
    # plotting.fitness_trend(ga.average_fitness, ga.best_fitness)
    # plotting.biodiversity_trend(ga.biodiversity)

    # timing
    # plotting.timing(ga.timings)

    # for k in ga.timings.keys():
    #     print(f"{k}: {ga.timings[k]:.3f} seconds")
    # print(f"total time: {sum(ga.timings.values()):.3f} seconds")
