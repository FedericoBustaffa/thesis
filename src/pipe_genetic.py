import os
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
        workers_num: int = os.cpu_count(),
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
        self.__population = self._generator.perform()
        self.__scores = self._evaluator.perform(self.__population)

        for g in range(max_generations):
            logger.debug(f"generation: {g + 1}")
            chosen = self._selector.perform(self.__population, self.__scores)
            couples = self._mater.perform(chosen)

            # parallel work
            for w in self.__workers:
                w.send(couples)

            offsprings = []
            offsprings_scores = []
            for w in self.__workers:
                offsprings_chunk, offsprings_scores_chunk = w.recv()
                offsprings.extend(offsprings_chunk)
                offsprings_scores.extend(offsprings_scores_chunk)

            self.__population, self.__scores = self._replacer.perform(
                self.__population, self.__scores, offsprings, offsprings_scores
            )

        for w in self.__workers:
            w.send(None)
            w.join()


if __name__ == "__main__":
    import sys
    import time
    from functools import partial

    import numpy as np
    import pandas as pd
    from loguru import logger

    import tsp
    from utils import plotting

    logger.remove()
    logger.add(
        sink=sys.stdout,
        colorize=True,
        level="INFO",
        format="<level>{level}: {message}</level>",
    )

    if len(sys.argv) < 7:
        logger.error(f"USAGE: py {sys.argv[0]} <T> <N> <G> <C> <M> <W>")
        exit(1)

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

    start = time.perf_counter()
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
    logger.success("solver ready")
    ga.run(G)
    logger.success(f"algorithm total time: {time.perf_counter() - start} seconds")

    best, best_score = ga.get()
    logger.success(f"best score: {best_score:.3f}")

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
