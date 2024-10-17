import multiprocessing as mp
import multiprocessing.queues as mpq
import queue
import threading
import time

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


def work(rqueue: mpq.Queue, squeue: mpq.Queue, toolbox: ToolBox, stats: Statistics):
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
            start = time.process_time()
            offsprings = toolbox.crossover(parents)
            stats.add_time("crossover", start)

            start = time.process_time()
            offsprings = toolbox.mutate(offsprings)
            stats.add_time("mutation", start)

            start = time.process_time()
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
        super().__init__(target=work, args=[self.rqueue, self.squeue, toolbox, stats])

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

    def join(self, timeout: float | None = None) -> None:
        self.squeue.close()
        self.rqueue.put(None)
        while not self.rqueue.empty():
            pass
        self.rqueue.close()
        super().join(timeout)
