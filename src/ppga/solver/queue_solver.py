import asyncio
import math
import multiprocessing as mp
import multiprocessing.queues as mpq
import time

from tqdm import tqdm

from ppga.base.statistics import Statistics
from ppga.base.toolbox import ToolBox
from ppga.solver.genetic_solver import GeneticSolver


def task(rqueue: mpq.Queue, squeue: mpq.Queue, toolbox: ToolBox, stats: Statistics):
    couples = []
    while True:
        couples = rqueue.get()

        if couples is None:
            break

        start = time.perf_counter()
        offsprings = toolbox.crossover(couples)
        stats.add_time("crossover", start)

        start = time.perf_counter()
        offsprings = toolbox.mutate(offsprings)
        stats.add_time("mutation", start)

        start = time.perf_counter()
        offsprings = toolbox.evaluate(offsprings)
        stats.add_time("evaluation", start)

        squeue.put((offsprings, stats.timings))


class QueueWorker(mp.Process):
    def __init__(self, toolbox: ToolBox, stats: Statistics) -> None:
        self.__rqueue = mp.Queue()
        self.__squeue = mp.Queue()
        super().__init__(
            target=task, args=[self.__rqueue, self.__squeue, toolbox, stats]
        )

    async def send(self, msg) -> None:
        self.__rqueue.put(msg)

    async def recv(self):
        return self.__squeue.get()

    def join(self, timeout: float | None = None):
        self.__rqueue.close()
        self.__squeue.close()
        super().join(timeout)


class QueuedGeneticSolver(GeneticSolver):
    def __init__(self, workers_num: int) -> None:
        self.workers_num = workers_num

    async def solve(
        self,
        toolbox: ToolBox,
        stats: Statistics,
        population_size: int,
        max_generations: int,
    ):
        # start the parallel workers
        workers = [QueueWorker(toolbox, stats) for _ in range(self.workers_num)]
        for w in workers:
            w.start()

        start = time.perf_counter()
        population = toolbox.generate(population_size)
        stats.add_time("generation", start)

        # start = time.perf_counter()
        population = toolbox.evaluate(population)
        # stats.add_time("evaluation", start)

        for g in tqdm(range(max_generations), desc="generations", ncols=80):
            start = time.perf_counter()
            chosen = toolbox.select(population)
            stats.add_time("selection", start)

            start = time.perf_counter()
            couples = toolbox.mate(chosen)
            stats.add_time("mating", start)

            # parallel work
            chunksize = math.ceil(len(couples) / len(workers))
            offsprings = []

            # sending couples chunks
            start = time.perf_counter()
            tasks = [
                asyncio.create_task(
                    workers[i].send(couples[i * chunksize : i * chunksize + chunksize])
                )
                for i in range(len(workers))
            ]
            asyncio.as_completed(tasks)

            # receiving offsprings and scores
            tasks = [asyncio.create_task(w.recv()) for w in workers]
            results = [await t for t in tasks]
            stats.add_time("parallel", start)
            for offsprings_chunk, timings in results:
                offsprings.extend(offsprings_chunk)
                if stats.timings["crossover"] < timings["crossover"]:
                    stats.timings["crossover"] += timings["crossover"]
                if stats.timings["mutation"] < timings["mutation"]:
                    stats.timings["mutation"] += timings["mutation"]
                if stats.timings["evaluation"] < timings["evaluation"]:
                    stats.timings["evaluation"] += timings["evaluation"]

            start = time.perf_counter()
            population = toolbox.replace(population, offsprings)
            stats.add_time("replacement", start)

            stats.push_best(population[0].fitness.fitness)
            stats.push_worst(population[-1].fitness.fitness)

        for w in workers:
            await asyncio.create_task(w.send(None))
            w.join()

        return population, stats

    def run(
        self,
        toolbox: ToolBox,
        population_size: int,
        max_generations: int,
        stats: Statistics,
    ):
        return asyncio.run(self.solve(toolbox, stats, population_size, max_generations))
