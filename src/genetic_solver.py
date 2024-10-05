from loguru import logger

import modules


class GeneticSolver:
    def set_chromosome(self, attribute_type=int, structure=list):
        self.attribute_type = attribute_type
        self.chromosome_structure = structure

    def set_fitness(self, weights: tuple):
        self.weights = weights

    def set_population(self, structure=list, *args, **kwargs):
        self.pop_structure = structure
        self.pop_args = args
        self.pop_kwargs = kwargs

    def set_generation(self, generation_func, *args, **kwargs):
        self.generation_func = generation_func
        self.generation_args = args
        self.generation_kwargs = kwargs

    def generate(self, population_size):
        return [
            self.generation_func(*self.generation_args, **self.generation_kwargs)
            for _ in range(population_size)
        ]

    def run(self, max_generations: int):
        self._population = self._generator.perform()
        self._scores = self._evaluator.perform(self._population)

        timing = 0.0
        for g in range(max_generations):
            logger.trace(f"generation: {g + 1}")

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

        logger.info(f"to parallelize time: {timing:.6f} seconds")

    def get(self, k: int = 1):
        if k > 1:
            return self._population[:k], self._scores[:k]
        else:
            return self._population[0], self._scores[0]


if __name__ == "__main__":
    import sys
    import time

    import numpy as np
    import pandas as pd

    import tsp
    from utils import plotting

    if len(sys.argv) < 7:
        logger.error(f"USAGE: py {sys.argv[0]} <T> <N> <G> <C> <M> <log_level>")
        exit(1)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | {file}:{line} | <level>{level} - {message}</level>",
        level=sys.argv[6].upper(),
        enqueue=True,
    )

    data = pd.read_csv(f"datasets/towns_{sys.argv[1]}.csv")
    towns = [tsp.Town(data["x"].iloc[i], data["y"].iloc[i]) for i in range(len(data))]

    # Initial population size
    N = int(sys.argv[2])

    # Max generations
    G = int(sys.argv[3])

    # crossover rate
    CR = float(sys.argv[4])

    # mutation rate
    MR = float(sys.argv[5])

    solver = GeneticSolver()
    solver.set_chromosome(attribute_type=int, structure=list)
    solver.set_fitness(weights=(1.0,))

    solver.set_generation(tsp.generate, len(towns))

    population = solver.generate(N)
    for i in population:
        logger.trace(f"{i}")

    solver.set_fitness(tsp.fitness, towns)
    # solver.set_selection(tsp.tournament)
    # solver.set_mating(tsp.couples_mating)
    # solver.set_crossover(tsp.one_point_no_rep, CR)
    # solver.set_mutation(tsp.rotation, MR)
    # solver.set_replacement(tsp.merge)

    # start = time.perf_counter()
    # hall_of_fame = solver.run(G, 5)
    # logger.info(f"total time: {time.perf_counter() - start:.6f} seconds")

    # logger.info(f"best score: {best.fitness:.6f}")

    # # drawing the graph
    # plotting.draw_graph(data, best)

    # statistics data
    # plotting.fitness_trend(ga.average_fitness, ga.best_fitness)
    # plotting.biodiversity_trend(ga.biodiversity)

    # timing
    # plotting.timing(ga.timings)

    # for k in ga.timings.keys():
    #     print(f"{k}: {ga.timings[k]:.3f} seconds")
    # print(f"total time: {sum(ga.timings.values()):.3f} seconds")
