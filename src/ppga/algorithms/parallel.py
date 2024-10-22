import os
import queue
import threading
import time

from tqdm import tqdm

from ppga.algorithms.queue_worker import QueueWorker
from ppga.base.hall_of_fame import HallOfFame
from ppga.base.statistics import Statistics
from ppga.base.toolbox import ToolBox


def handle_worker(
    send_buffer: queue.Queue,
    recv_buffer: queue.Queue,
    toolbox: ToolBox,
    stats: Statistics,
):
    worker = QueueWorker(toolbox, stats)
    while True:
        chunk = send_buffer.get()
        send_buffer.task_done()
        if chunk is None:
            break

        worker.send(chunk)
        recv_buffer.put(worker.recv())
    worker.join()


class Handler(threading.Thread):
    def __init__(self, toolbox, stats):
        self.send_buffer = queue.Queue()
        self.recv_buffer = queue.Queue()
        super().__init__(
            target=handle_worker,
            args=[self.send_buffer, self.recv_buffer, toolbox, stats],
        )

    def send(self, chunk):
        self.send_buffer.put(chunk)

    def recv(self):
        obj = self.recv_buffer.get()
        self.recv_buffer.task_done()

        return obj

    def join(self, timeout: float | None = None) -> None:
        self.send_buffer.put(None)
        self.send_buffer.join()
        self.recv_buffer.join()
        super().join(timeout)


def generational(
    toolbox: ToolBox,
    population_size: int,
    max_generations: int,
    hall_of_fame: None | HallOfFame = None,
):
    stats = Statistics()

    # start the parallel workers
    workers_num = 1
    assert workers_num is not None

    handlers = [Handler(toolbox, stats) for _ in range(workers_num)]
    for handler in handlers:
        handler.start()

    start = time.perf_counter()
    population = toolbox.generate(population_size)
    stats.add_time("generation", start)

    for g in tqdm(range(max_generations), desc="generations", ncols=80):
        start = time.perf_counter()
        chosen = toolbox.select(population, population_size)
        stats.add_time("selection", start)

        start = time.perf_counter()
        couples = toolbox.mate(chosen)
        stats.add_time("mating", start)

        # parallel crossover + mutation + evaluation
        chunksize = len(couples) // workers_num
        carry = len(couples) % workers_num

        start = time.perf_counter()
        for i in range(carry):
            handlers[i].send(couples[i * chunksize : i * chunksize + chunksize + 1])

        for i in range(carry, workers_num, 1):
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

        stats["crossover"] += crossover_time
        stats["mutation"] += mutation_time
        stats["evaluation"] += evaluation_time
        stats.add_time("parallel", start)

        # replacement
        start = time.perf_counter()
        population = toolbox.replace(population, offsprings)
        stats.add_time("replacement", start)

        stats.push_best(max(population).fitness)
        stats.push_worst(min([i for i in population if not i.invalid()]).fitness)

        if hall_of_fame is not None:
            hall_of_fame.update(population)

    for h in handlers:
        h.join()

    return population, stats
