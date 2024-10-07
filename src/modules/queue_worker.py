import multiprocessing as mp
import multiprocessing.queues as mpq

from loguru import logger

from modules.toolbox import ToolBox


def task(
    rqueue: mpq.Queue,
    squeue: mpq.Queue,
    toolbox: ToolBox,
):
    logger.trace(f"{mp.current_process().name} started")
    couples = []
    while True:
        couples = rqueue.get()

        if couples is None:
            break

        offsprings = toolbox.crossover(couples)
        offsprings = toolbox.mutate(offsprings)
        offsprings = toolbox.evaluate(offsprings)

        squeue.put(offsprings)

    logger.trace(f"{mp.current_process().name} terminated")


class QueueWorker(mp.Process):
    def __init__(self, toolbox: ToolBox) -> None:
        self.__rqueue = mp.Queue()
        self.__squeue = mp.Queue()
        super().__init__(target=task, args=[self.__rqueue, self.__squeue, toolbox])

    def send(self, msg) -> None:
        self.__rqueue.put(msg)

    def recv(self):
        return self.__squeue.get()

    def join(self):
        self.__rqueue.close()
        self.__squeue.close()
        super().join()


if __name__ == "__main__":
    pass
    pass
    pass
