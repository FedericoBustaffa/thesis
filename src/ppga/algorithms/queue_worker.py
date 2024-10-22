import multiprocessing as mp
import multiprocessing.queues as mpq
import queue
import threading
import time

from loguru import logger

from ppga.base import Statistics, ToolBox


def get(rqueue: mpq.Queue, local_queue: queue.Queue):
    while True:
        chunk = rqueue.get()
        local_queue.put(chunk)
        if chunk is None:
            break

        while chunk is not None:
            chunk = rqueue.get()
            local_queue.put(chunk)

    logger.debug("getter done")


def work(rqueue: mpq.Queue, squeue: mpq.Queue, toolbox: ToolBox, stats: Statistics):
    local_queue = queue.Queue()

    # getter thread
    getter = threading.Thread(target=get, args=[rqueue, local_queue])
    getter.start()

    while True:
        parents = local_queue.get()
        local_queue.task_done()
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
            local_queue.task_done()

        squeue.put(stats.timings)

    local_queue.join()
    getter.join()
    logger.debug(f"{mp.current_process().name} done")


class QueueWorker:
    def __init__(self, toolbox: ToolBox, stats: Statistics) -> None:
        self.lock = mp.Lock()
        self.rqueue = mp.Queue()
        self.squeue = mp.Queue()
        self.worker = mp.Process(
            target=work, args=[self.rqueue, self.squeue, toolbox, stats], daemon=False
        )
        self.worker.start()

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

    def join(self) -> None:
        if self.worker.is_alive():
            if not self.rqueue.empty():
                logger.debug("wait for the recv queue to be empty")
            self.rqueue.put(None)
            self.rqueue.close()
            self.rqueue.join_thread()

            if not self.squeue.empty():
                logger.debug("wait for the send queue to be empty")
            self.squeue.close()
            self.squeue.join_thread()

            logger.debug(f"{self.worker.name} joining")
            self.worker.join()
            logger.debug(f"{self.worker.name} joined")
