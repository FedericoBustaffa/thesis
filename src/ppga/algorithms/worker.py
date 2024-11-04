import multiprocessing as mp
import multiprocessing.queues as mpq

from ppga import log
from ppga.algorithms.reproduction import reproduction
from ppga.base.toolbox import ToolBox


def work(
    rqueue: mpq.Queue,
    squeue: mpq.Queue,
    toolbox: ToolBox,
    lam: int,
    cxpb: float,
    mutpb: float,
):
    logger = log.core_logger
    while True:
        parents = squeue.get()
        if parents is None:
            break

        if len(parents) == 0:
            logger.warning(f"worker unused: {len(parents)} parents given")
            # continue

        offsprings = reproduction(parents, toolbox, lam, cxpb, mutpb)

        for offspring in offsprings:
            offspring = toolbox.evaluate(offspring)

        rqueue.put(offsprings)


class Worker:
    def __init__(self, toolbox: ToolBox, lam: int, cxpb: float, mutpb: float) -> None:
        self.rqueue = mp.Queue()
        self.squeue = mp.Queue()
        self.worker = mp.Process(
            target=work, args=[self.rqueue, self.squeue, toolbox, lam, cxpb, mutpb]
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
