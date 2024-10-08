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
    while True:
        parents = rqueue.get()
        if parents is None:
            break

        while parents is not None:
            start = time.perf_counter()
            offsprings = toolbox.crossover(parents)
            stats.add_time("crossover", start)

            start = time.perf_counter()
            offsprings = toolbox.mutate(offsprings)
            stats.add_time("mutation", start)

            start = time.perf_counter()
            offsprings = toolbox.evaluate(offsprings)
            stats.add_time("evaluation", start)

            squeue.put(offsprings)
            parents = rqueue.get()

        squeue.put(stats.timings)


class QueueWorker(mp.Process):
    def __init__(self, toolbox: ToolBox, stats: Statistics) -> None:
        self.__rqueue = mp.Queue()
        self.__squeue = mp.Queue()
        super().__init__(
            target=task, args=[self.__rqueue, self.__squeue, toolbox, stats]
        )

    async def send(self, chunk: list | None = None) -> None:
        if isinstance(chunk, list):
            size = len(chunk) // 8
            for i in range(8):
                self.__rqueue.put(chunk[i * size : i * size + size])
        self.__rqueue.put(None)

    async def recv(self):
        obj = self.__squeue.get()
        if isinstance(obj, list):
            result = []
            while not isinstance(obj, dict):
                result.extend(obj)
                obj = self.__squeue.get()
            return (result, obj)
        else:
            return obj

    def join(self, timeout: float | None = None):
        self.__squeue.close()
        self.__rqueue.put(None)
        while not self.__rqueue.empty():
            pass
        self.__rqueue.close()
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

        # this one should not be timed
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

            # parallel crossover + mutation + evaluation
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

            # keep only the worst time for each worker
            for offsprings_chunk, timings in results:
                offsprings.extend(offsprings_chunk)
                if stats.timings["crossover"] < timings["crossover"]:
                    stats.timings["crossover"] += timings["crossover"]
                if stats.timings["mutation"] < timings["mutation"]:
                    stats.timings["mutation"] += timings["mutation"]
                if stats.timings["evaluation"] < timings["evaluation"]:
                    stats.timings["evaluation"] += timings["evaluation"]

            # replacement
            start = time.perf_counter()
            population = toolbox.replace(population, offsprings)
            stats.add_time("replacement", start)

            stats.push_best(population[0].fitness.fitness)
            stats.push_worst(population[-1].fitness.fitness)

        for w in workers:
            w.join()

        return population, stats

    def run(
        self,
        toolbox: ToolBox,
        stats: Statistics,
        population_size: int,
        max_generations: int,
    ):
        return asyncio.run(self.solve(toolbox, stats, population_size, max_generations))
