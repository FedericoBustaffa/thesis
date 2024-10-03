import multiprocessing as mp
import multiprocessing.queues as mpq

from loguru import logger

from modules import Crossoverator, Evaluator, Mutator


def qtask(
    rqueue: mpq.Queue,
    squeue: mpq.Queue,
    crossoverator: Crossoverator,
    mutator: Mutator,
    evaluator: Evaluator,
):
    logger.trace(f"{mp.current_process().name} started")
    while True:
        couples = rqueue.get()
        if couples is None:
            break

        offsprings = crossoverator.perform(couples)
        offsprings = mutator.perform(offsprings)
        scores = evaluator.perform(offsprings)

        squeue.put_nowait((offsprings, scores))

    logger.trace(f"{mp.current_process().name} terminated")


class QueueWorker:
    def __init__(self, crossoverator, mutator, evaluator) -> None:
        self.__rqueue = mp.Queue()
        self.__squeue = mp.Queue()
        self.__process = mp.Process(
            target=qtask,
            args=[self.__rqueue, self.__squeue, crossoverator, mutator, evaluator],
        )

    def start(self) -> None:
        self.__process.start()

    def send(self, msg) -> None:
        self.__rqueue.put_nowait(msg)

    def recv(self):
        return self.__squeue.get()

    def join(self) -> None:
        self.__rqueue.close()
        self.__squeue.close()
        self.__process.join()


if __name__ == "__main__":
    pass
