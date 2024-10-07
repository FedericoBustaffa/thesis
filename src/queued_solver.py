import math
import time

from loguru import logger

from genetic_solver import GeneticSolver
from modules import Individual, QueueWorker, ToolBox


class QueuedGeneticSolver(GeneticSolver):
    def __init__(self, workers_num: int) -> None:
        self.workers_num = workers_num

    def run(
        self, toolbox: ToolBox, population_size: int, max_generations: int
    ) -> list[Individual]:
        workers = [QueueWorker(toolbox) for _ in range(self.workers_num)]

        for w in workers:
            w.start()

        population = toolbox.generate(population_size)
        population = toolbox.evaluate(population)

        timing = 0.0
        send_time = 0.0
        for g in range(max_generations):
            logger.trace(f"generation: {g + 1}")

            chosen = toolbox.select(population)
            couples = toolbox.mate(chosen)

            # parallel work
            offsprings = []
            chunksize = math.ceil(len(couples) / len(workers))

            # sending couples chunks
            start = time.perf_counter()
            send_start = time.perf_counter()
            for i in range(len(workers)):
                workers[i].send(couples[i * chunksize : i * chunksize + chunksize])
            send_time += time.perf_counter() - send_start

            # receiving offsprings and scores
            results = [w.recv() for w in workers]
            for offsprings_chunk in results:
                offsprings.extend(offsprings_chunk)
            timing += time.perf_counter() - start

            population = toolbox.replace(population, offsprings)

        for w in workers:
            w.send(None)
            w.join()

        logger.info(f"parallel time: {timing} seconds")
        logger.info(f"send time: {send_time:.6f} seconds")

        return population
