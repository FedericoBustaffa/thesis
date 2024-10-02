import multiprocessing as mp
import multiprocessing.connection as conn

from loguru import logger

from modules import Crossoverator, Evaluator, Mutator


def task(
    pipe: conn.Connection,
    crossoverator: Crossoverator,
    mutator: Mutator,
    evaluator: Evaluator,
):
    logger.trace(f"{mp.current_process().name} started")
    while True:
        couples = pipe.recv()
        if couples is None:
            break

        offsprings = crossoverator.perform(couples)
        offsprings = mutator.perform(offsprings)
        scores = evaluator.perform(offsprings)

        pipe.send((offsprings, scores))

    pipe.close()
    logger.trace(f"{mp.current_process().name} terminated")


class Worker:
    def __init__(self, crossoverator, mutator, evaluator) -> None:
        self.__pipe, process_pipe = mp.Pipe()
        self.__process = mp.Process(
            target=task, args=[process_pipe, crossoverator, mutator, evaluator]
        )

    def start(self) -> None:
        self.__process.start()

    async def send(self, msg) -> None:
        self.__pipe.send(msg)

    async def recv(self):
        return self.__pipe.recv()

    def join(self) -> None:
        self.__pipe.close()
        self.__process.join()


if __name__ == "__main__":
    pass
