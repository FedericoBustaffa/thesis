import math
import queue
import threading
import time

from tqdm import tqdm

from ppga.algorithms.queue_worker import QueueWorker
from ppga.base.statistics import Statistics
from ppga.base.toolbox import ToolBox


def handle_worker(
    send_buffer: queue.Queue, recv_buffer: queue.Queue, worker: QueueWorker
):
    worker.start()
    while True:
        chunk = send_buffer.get()
        if chunk is None:
            break

        worker.send(chunk)
        recv_buffer.put(worker.recv())
    worker.join()


class Handler(threading.Thread):
    def __init__(self, worker: QueueWorker):
        self.send_buffer = queue.Queue()
        self.recv_buffer = queue.Queue()
        super().__init__(
            target=handle_worker, args=[self.send_buffer, self.recv_buffer, worker]
        )

    def send(self, chunk):
        self.send_buffer.put(chunk)

    def recv(self):
        return self.recv_buffer.get()

    def join(self, timeout: float | None = None) -> None:
        self.send_buffer.put(None)
        super().join(timeout)


def generational(
    toolbox: ToolBox,
    population_size: int,
    max_generations: int,
    workers_num: int,
):
    stats = Statistics()

    # start the parallel workers
    workers = [QueueWorker(toolbox, stats) for _ in range(workers_num)]
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

    for g in tqdm(range(max_generations), desc="generations", ncols=80):
        start = time.perf_counter()
        chosen = toolbox.select(population)
        stats.add_time("selection", start)

        start = time.perf_counter()
        couples = toolbox.mate(chosen)
        stats.add_time("mating", start)

        # parallel crossover + mutation + evaluation
        chunksize = len(couples) // workers_num
        carry = len(couples) % workers_num

        start = time.perf_counter()
        for i in range(len(handlers)):
            if carry > 0:
                handlers[i].send(couples[i * chunksize : i * chunksize + chunksize + 1])
                carry -= 1
            else:
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

        stats.timings["crossover"] += crossover_time
        stats.timings["mutation"] += mutation_time
        stats.timings["evaluation"] += evaluation_time
        stats.add_time("parallel", start)

        # replacement
        start = time.perf_counter()
        population = toolbox.replace(population, offsprings)
        stats.add_time("replacement", start)

        stats.push_best(population[0].fitness.fitness)
        stats.push_worst(population[-1].fitness.fitness)

    for h in handlers:
        h.join()

    return population, stats
