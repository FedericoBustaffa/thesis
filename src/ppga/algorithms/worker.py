import multiprocessing as mp
import multiprocessing.queues as mpq

from ppga.base import ToolBox


def work(rqueue: mpq.Queue, squeue: mpq.Queue, toolbox: ToolBox):
    while True:
        parents = squeue.get()
        if parents is None:
            break

        offsprings = toolbox.crossover(parents)
        offsprings = toolbox.mutate(offsprings)
        offsprings = toolbox.evaluate(offsprings)

        rqueue.put(offsprings)


class Worker:
    def __init__(self, toolbox: ToolBox) -> None:
        self.lock = mp.Lock()
        self.rqueue = mp.Queue()
        self.squeue = mp.Queue()
        self.worker = mp.Process(target=work, args=[self.rqueue, self.squeue, toolbox])
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
