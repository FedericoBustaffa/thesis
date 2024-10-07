import time

from loguru import logger

from ppga.base.individual import Individual
from ppga.base.statistics import Statistics
from ppga.base.toolbox import ToolBox


class GeneticSolver:
    def run(
        self, toolbox: ToolBox, stats: Statistics, population_size, max_generations: int
    ) -> tuple[list[Individual], Statistics]:
        population = toolbox.generate(population_size)
        population = toolbox.evaluate(population)

        parallel_time = 0.0
        for g in range(max_generations):
            logger.trace(f"generation: {g + 1}")

            chosen = toolbox.select(population)
            couples = toolbox.mate(chosen)

            start = time.perf_counter()
            offsprings = toolbox.crossover(couples)
            offsprings = toolbox.mutate(offsprings)
            offsprings = toolbox.evaluate(offsprings)
            parallel_time += time.perf_counter() - start

            population = toolbox.replace(population, offsprings)

            stats.push_best(population[0].fitness.fitness)
            stats.push_worst(population[-1].fitness.fitness)

        stats.add_time("parallel", parallel_time)

        return population, stats
