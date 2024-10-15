import asyncio
import math
import multiprocessing as mp
import multiprocessing.queues as mpq
import queue
import threading
import time

from tqdm import tqdm

from ppga.base.statistics import Statistics
from ppga.base.toolbox import ToolBox
from ppga.solver.genetic_solver import GeneticSolver


def get(rqueue: mpq.Queue, local_queue: queue.Queue):
    while True:
        chunk = rqueue.get()
        local_queue.put(chunk)
        if chunk is None:
            break

        while chunk is not None:
            chunk = rqueue.get()
            local_queue.put(chunk)


def task(rqueue: mpq.Queue, squeue: mpq.Queue, toolbox: ToolBox, stats: Statistics):
    local_queue = queue.Queue()

    # getter thread
    getter = threading.Thread(target=get, args=[rqueue, local_queue])
    getter.start()

    while True:
        parents = local_queue.get()
        if parents is None:
            break

        stats.reset()
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
            parents = local_queue.get()

        squeue.put(stats.timings)

    getter.join()


class QueueWorker(mp.Process):
    def __init__(self, toolbox: ToolBox, stats: Statistics) -> None:
        self.__rqueue = mp.Queue()
        self.__squeue = mp.Queue()
        super().__init__(
            target=task, args=[self.__rqueue, self.__squeue, toolbox, stats]
        )

        self.subchunks = 2

    async def send(self, chunk: list | None = None) -> None:
        if isinstance(chunk, list):
            size = len(chunk) // self.subchunks
            for i in range(self.subchunks):
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
        population_size: int,
        max_generations: int,
        stats: Statistics,
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

            # keep only the worst time for each worker
            offsprings = []
            crossover_time = 0.0
            mutation_time = 0.0
            evaluation_time = 0.0
            for offsprings_chunk, timings in results:
                offsprings.extend(offsprings_chunk)

                if crossover_time < timings["crossover"]:
                    crossover_time = timings["crossover"]

                if mutation_time < timings["mutation"]:
                    mutation_time = timings["mutation"]

                if evaluation_time < timings["evaluation"]:
                    evaluation_time = timings["evaluation"]
            stats.add_time("parallel", start)

            stats.timings["crossover"] += crossover_time
            stats.timings["mutation"] += mutation_time
            stats.timings["evaluation"] += evaluation_time

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
        population_size: int,
        max_generations: int,
        stats: Statistics,
    ):
        loop = asyncio.get_event_loop()
        task = loop.create_task(self.solve(toolbox, population_size, max_generations, stats))
        result = asyncio.as_completed(task)

        return result.result()
