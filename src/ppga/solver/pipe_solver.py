import asyncio
import math
import multiprocessing as mp
import multiprocessing.connection as conn
import time

from loguru import logger

from ppga.base.statistics import Statistics
from ppga.base.toolbox import ToolBox
from ppga.solver.genetic_solver import GeneticSolver


def task(pipe: conn.Connection, toolbox: ToolBox):
    while True:
        couples = pipe.recv()
        if couples is None:
            break

        offsprings = toolbox.crossover(couples)
        offsprings = toolbox.mutate(offsprings)
        offsprings = toolbox.evaluate(offsprings)

        pipe.send(offsprings)

    pipe.close()


class PipeWorker(mp.Process):
    def __init__(self, toolbox) -> None:
        self.__pipe, process_pipe = mp.Pipe()
        super().__init__(target=task, args=[process_pipe, toolbox])

    async def send(self, msg) -> None:
        self.__pipe.send(msg)

    async def recv(self):
        return self.__pipe.recv()

    def join(self, timeout: float | None = None):
        self.__pipe.close()
        super().join(timeout)


class PipeGeneticSolver(GeneticSolver):
    def __init__(self, workers_num: int) -> None:
        self.workers_num = workers_num

    async def solve(
        self,
        toolbox: ToolBox,
        population_size: int,
        max_generations: int,
        stats: Statistics,
    ):
        workers = [PipeWorker(toolbox) for _ in range(self.workers_num)]
        for w in workers:
            w.start()

        population = toolbox.generate(population_size)
        population = toolbox.evaluate(population)

        parallel_time = 0.0
        send_time = 0.0
        for g in range(max_generations):
            logger.trace(f"generation: {g + 1}")

            chosen = toolbox.select(population)
            couples = toolbox.mate(chosen)

            # parallel work
            chunksize = math.ceil(len(couples) / len(workers))
            offsprings = []

            # sending couples chunks
            start = time.perf_counter()
            send_start = time.perf_counter()
            tasks = [
                asyncio.create_task(
                    workers[i].send(couples[i * chunksize : i * chunksize + chunksize])
                )
                for i in range(len(workers))
            ]
            asyncio.as_completed(tasks)
            send_time += time.perf_counter() - send_start

            # receiving offsprings and scores
            tasks = [asyncio.create_task(w.recv()) for w in workers]
            results = [await t for t in tasks]
            for offsprings_chunk in results:
                offsprings.extend(offsprings_chunk)

            parallel_time += time.perf_counter() - start

            population = toolbox.replace(population, offsprings)
            stats.push_best(population[0].fitness.fitness)
            stats.push_worst(population[-1].fitness.fitness)

        for w in workers:
            await asyncio.create_task(w.send(None))
            w.join()

        stats.add_time("parallel", parallel_time)
        stats.add_time("synchronization", send_time)

        return population, stats

    def run(
        self,
        toolbox: ToolBox,
        population_size: int,
        max_generations: int,
        stats: Statistics,
    ):
        return asyncio.run(self.solve(toolbox, population_size, max_generations, stats))
