import multiprocessing as mp
import multiprocessing.queues as mpq
import time

import numpy as np

from ppga import log
from ppga.algorithms.reproduction import reproduction
from ppga.base.toolbox import ToolBox


def task(
    rqueue: mpq.Queue,
    squeue: mpq.Queue,
    toolbox: ToolBox,
    cxpb: float,
    mutpb: float,
    log_level: str | int = log.INFO,
):
    eval_times = []
    logger = log.getCoreLogger(log_level)
    logger.debug(f"{mp.current_process().name} start")
    while True:
        parents = squeue.get()
        if parents is None:
            logger.debug(f"{mp.current_process().name} closing")
            break

        offsprings = reproduction(parents, toolbox, cxpb, mutpb)

        for i in range(len(offsprings)):
            if offsprings[i].invalid:
                eval_start = time.perf_counter()
                offsprings[i] = toolbox.evaluate(offsprings[i])
                eval_times.append(time.perf_counter() - eval_start)

        rqueue.put(offsprings)

    logger.info(
        f"{mp.current_process().name} eval mean time: {np.mean(eval_times):.4f} seconds"
    )


class Worker:
    def __init__(
        self,
        toolbox: ToolBox,
        cxpb: float,
        mutpb: float,
        log_level: str | int = log.INFO,
    ) -> None:
        self.rqueue = mp.Queue()
        self.squeue = mp.Queue()

        self.worker = mp.Process(
            target=task,
            args=[self.rqueue, self.squeue, toolbox, cxpb, mutpb, log_level],
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
