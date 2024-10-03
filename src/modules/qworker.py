import multiprocessing as mp
import multiprocessing.queues as mpq

from loguru import logger

from modules import Crossoverator, Evaluator, Mutator


def qtask(
    queue: mpq.Queue,
    crossoverator: Crossoverator,
    mutator: Mutator,
    evaluator: Evaluator,
):
    logger.trace(f"{mp.current_process().name} started")
    while True:
        couples = queue.get()
        queue.join_thread()
        for c in couples:
            logger.debug(f"{mp.current_process().name}: {c}")

        if couples is None:
            break

        offsprings = crossoverator.perform(couples)
        offsprings = mutator.perform(offsprings)
        scores = evaluator.perform(offsprings)

        queue.put((offsprings, scores))

    logger.trace(f"{mp.current_process().name} terminated")


class QueueWorker:
    def __init__(self, crossoverator, mutator, evaluator) -> None:
        self.__queue = mp.Queue(1)
        self.__process = mp.Process(
            target=qtask, args=[self.__queue, crossoverator, mutator, evaluator]
        )

    def start(self) -> None:
        self.__process.start()

    def send(self, msg) -> None:
        self.__queue.put(msg)

    def recv(self):
        return self.__queue.get()

    def join(self) -> None:
        self.__queue.close()
        self.__process.join()


if __name__ == "__main__":
    pass
