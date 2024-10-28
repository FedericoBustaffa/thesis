import multiprocessing as mp
import multiprocessing.queues as mpq
import time

from ppga.algorithms import simple
from ppga.base.toolbox import ToolBox


def work(
    rqueue: mpq.Queue, squeue: mpq.Queue, toolbox: ToolBox, cxpb: float, mutpb: float
):
    worker_file = open(f"./results/{mp.current_process().name}.txt", "w")
    while True:
        parents = squeue.get()
        if parents is None:
            break

        offsprings = simple.reproduction(parents, cxpb, mutpb, toolbox)

        for offspring in offsprings:
            start = time.perf_counter()
            offspring = toolbox.evaluate(offspring)
            eval_time = time.perf_counter() - start
            print(eval_time, file=worker_file)

        rqueue.put(offsprings)

    worker_file.close()


class Worker:
    def __init__(self, toolbox: ToolBox, cxpb: float, mutpb: float) -> None:
        self.rqueue = mp.Queue()
        self.squeue = mp.Queue()
        self.worker = mp.Process(
            target=work, args=[self.rqueue, self.squeue, toolbox, cxpb, mutpb]
        )
        self.worker.start()

    def send(self, chunk: list | None = None) -> None:
        self.squeue.put(chunk)

    def recv(self):
        return self.rqueue.get()

    def join(self) -> None:
        self.squeue.put(None)
        self.squeue.close()
        self.squeue.join_thread()

        self.rqueue.close()
        self.rqueue.join_thread()

        self.worker.join()
