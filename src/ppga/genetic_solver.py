import time

from loguru import logger

from ppga import Individual, ToolBox


class GeneticSolver:

    def run(
        self, toolbox: ToolBox, population_size, max_generations: int
    ) -> list[Individual]:
        population = toolbox.generate(population_size)
        population = toolbox.evaluate(population)

        timing = 0.0
        for g in range(max_generations):
            logger.trace(f"generation: {g + 1}")

            chosen = toolbox.select(population)
            couples = toolbox.mate(chosen)

            start = time.perf_counter()
            offsprings = toolbox.crossover(couples)
            offsprings = toolbox.mutate(offsprings)
            offsprings = toolbox.evaluate(offsprings)
            timing += time.perf_counter() - start

            population = toolbox.replace(population, offsprings)

        logger.info(f"to parallelize time: {timing:.6f} seconds")

        return population
