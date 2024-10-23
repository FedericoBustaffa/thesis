import multiprocessing as mp
import multiprocessing.queues as mpq
import time

from ppga.base import Statistics, ToolBox


def work(rqueue: mpq.Queue, squeue: mpq.Queue, toolbox: ToolBox, stats: Statistics):
    total_time = 0.0
    pure_work = 0.0
    times = []
    total_start = time.perf_counter()
    while True:
        parents = squeue.get()
        if parents is None:
            break

        stats.reset()
        start = time.perf_counter()
        offsprings = toolbox.crossover(parents)
        stats.add_time("crossover", start)
        pure_work += stats["crossover"]

        start = time.perf_counter()
        offsprings = toolbox.mutate(offsprings)
        stats.add_time("mutation", start)
        pure_work += stats["mutation"]

        start = time.perf_counter()
        offsprings, mean_time = toolbox.evaluate(offsprings)
        stats.add_time("evaluation", start)
        pure_work += stats["evaluation"]
        times.append(mean_time)

        rqueue.put((offsprings, stats.timings))
    total_time = time.perf_counter() - total_start

    name = mp.current_process().name
    print(f"{name} total time: {total_time} seconds")
    print(f"{name} pure work time: {pure_work} seconds")
    print(f"{name} eval mean time: {sum(times) / len(times)} seconds")


class Worker:
    def __init__(self, toolbox: ToolBox, stats: Statistics) -> None:
        self.lock = mp.Lock()
        self.rqueue = mp.Queue()
        self.squeue = mp.Queue()
        self.worker = mp.Process(
            target=work, args=[self.rqueue, self.squeue, toolbox, stats], daemon=False
        )
        self.worker.start()

    def send(self, chunk: list | None = None) -> None:
        self.squeue.put(chunk)

    def recv(self):
        return self.rqueue.get()

    def join(self) -> None:
        if self.worker.is_alive():
            self.squeue.put(None)
            self.squeue.close()
            self.squeue.join_thread()

            self.rqueue.close()
            self.rqueue.join_thread()

            self.worker.join()
