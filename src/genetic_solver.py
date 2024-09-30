import modules


class GeneticSolver:
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
    ) -> None:
        self._generator = modules.Generator(population_size, generation_func)
        self._evaluator = modules.Evaluator(fitness_func)
        self._selector = modules.Selector(selection_func)
        self._mater = modules.Mater(mating_func)
        self._crossoverator = modules.Crossoverator(crossover_func, crossover_rate)
        self._mutator = modules.Mutator(mutation_func, mutation_rate)
        self._replacer = modules.Replacer(replace_func)

    def run(self, max_generations: int):
        self._population = self._generator.perform()
        self._scores = self._evaluator.perform(self._population)

        timing = 0.0
        for g in range(max_generations):
            chosen = self._selector.perform(self._population, self._scores)
            couples = self._mater.perform(chosen)
            start = time.perf_counter()
            offsprings = self._crossoverator.perform(couples)
            offsprings = self._mutator.perform(offsprings)
            offsprings_scores = self._evaluator.perform(offsprings)
            timing += time.perf_counter() - start
            self._population, self._scores = self._replacer.perform(
                self._population, self._scores, offsprings, offsprings_scores
            )

        logger.info(f"{timing}")

    def get(self, k: int = 1):
        if k > 1:
            return self._population[:k], self._scores[:k]
        else:
            return self._population[0], self._scores[0]


if __name__ == "__main__":
    import sys
    import time
    from functools import partial

    import numpy as np
    import pandas as pd
    from loguru import logger

    import tsp
    from utils import plotting

    if len(sys.argv) < 6:
        logger.error(f"USAGE: py {sys.argv[0]} <T> <N> <G> <C> <M>")
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

    # partial functions to fix the arguments
    generate_func = partial(tsp.generate, len(towns))
    fitness_func = partial(tsp.fitness, towns)

    start = time.perf_counter()
    ga = GeneticSolver(
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
    )

    logger.debug("solver created")

    ga.run(G)
    logger.info(f"algorithm total time: {time.perf_counter() - start} seconds")

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
