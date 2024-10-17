import math
import multiprocessing as mp
import multiprocessing.queues as mpq
import queue
import threading
import time

from tqdm import tqdm

from ppga.base.statistics import Statistics
from ppga.base.toolbox import ToolBox


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
        self.rqueue = mp.Queue()
        self.squeue = mp.Queue()
        super().__init__(target=task, args=[self.rqueue, self.squeue, toolbox, stats])

        self.subchunks = 2

    def send(self, chunk: list | None = None) -> None:
        if isinstance(chunk, list):
            size = len(chunk) // self.subchunks
            for i in range(self.subchunks):
                self.rqueue.put(chunk[i * size : i * size + size])
        self.rqueue.put(None)

    def recv(self):
        obj = self.squeue.get()
        if isinstance(obj, list):
            result = []
            while not isinstance(obj, dict):
                result.extend(obj)
                obj = self.squeue.get()
            return (result, obj)
        else:
            return obj

    def join(self, timeout: float | None = None):
        self.squeue.close()
        self.rqueue.put(None)
        while not self.rqueue.empty():
            pass
        self.rqueue.close()
        super().join(timeout)


def handle_worker(buffer: queue.Queue, worker: QueueWorker):
    worker.start()
    while True:
        chunk = buffer.get()
        if chunk is None:
            break

        worker.send(chunk)
        result = worker.recv()
        buffer.put(result[0])
        buffer.put(result[1])


class Handler(threading.Thread):
    def __init__(self, worker: QueueWorker):
        self.buffer = queue.Queue()
        super().__init__(target=handle_worker, args=[self.buffer, worker])

    def send(self, chunk):
        self.buffer.put(chunk)

    def recv(self):
        return self.buffer.get()

    def join(self, timeout):
        self.buffer.put(None)
        super().join(timeout)


class QueuedGeneticSolver:
    def __init__(self, workers_num: int) -> None:
        self.workers_num = workers_num

    def run(
        self,
        toolbox: ToolBox,
        population_size: int,
        max_generations: int,
        stats: Statistics,
    ):
        # start the parallel workers
        workers = [QueueWorker(toolbox, stats) for _ in range(self.workers_num)]
        handlers = [Handler(w) for w in workers]
        for handler in handlers:
            handler.start()

        start = time.perf_counter()
        population = toolbox.generate(population_size)
        stats.add_time("generation", start)

        # this one should not be timed
        # start = time.perf_counter()
        population = toolbox.evaluate(population)
        # stats.add_time("evaluation", start)

        # for g in tqdm(range(max_generations), desc="generations", ncols=80):
        for g in range(max_generations):
            start = time.perf_counter()
            chosen = toolbox.select(population)
            stats.add_time("selection", start)

            start = time.perf_counter()
            couples = toolbox.mate(chosen)
            stats.add_time("mating", start)

            # parallel crossover + mutation + evaluation
            chunksize = math.ceil(len(couples) / len(workers))

            start = time.perf_counter()
            for i in range(len(handlers)):
                handlers[i].send(couples[i * chunksize : i * chunksize + chunksize])

            # keep only the worst time for each worker
            offsprings = []
            crossover_time = 0.0
            mutation_time = 0.0
            evaluation_time = 0.0

            for handler in handlers:
                offsprings_chunk, timings = handler.recv()

                offsprings.extend(offsprings_chunk)

                if crossover_time < timings["crossover"]:
                    crossover_time = timings["crossover"]

                if mutation_time < timings["mutation"]:
                    mutation_time = timings["mutation"]

                if evaluation_time < timings["evaluation"]:
                    evaluation_time = timings["evaluation"]

            stats.add_time("parallel", start)
            stats.add_time("crossover", crossover_time)
            stats.add_time("mutation", mutation_time)
            stats.add_time("evaluation", evaluation_time)

            # replacement
            start = time.perf_counter()
            population = toolbox.replace(population, offsprings)
            stats.add_time("replacement", start)

            stats.push_best(population[0].fitness.fitness)
            stats.push_worst(population[-1].fitness.fitness)

        for w in workers:
            w.join()

        for h in handlers:
            h.join()

        return population, stats
