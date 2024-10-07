import multiprocessing as mp
import multiprocessing.connection as conn

from loguru import logger

from modules.toolbox import ToolBox


def task(pipe: conn.Connection, toolbox: ToolBox):
    logger.trace(f"{mp.current_process().name} started")
    while True:
        couples = pipe.recv()
        if couples is None:
            break

        offsprings = toolbox.crossover(couples)
        offsprings = toolbox.mutate(offsprings)
        offsprings = toolbox.evaluate(offsprings)

        pipe.send(offsprings)

    pipe.close()
    logger.trace(f"{mp.current_process().name} terminated")


class PipeWorker(mp.Process):
    def __init__(self, toolbox) -> None:
        self.__pipe, process_pipe = mp.Pipe()
        super().__init__(target=task, args=[process_pipe, toolbox])

    async def send(self, msg) -> None:
        self.__pipe.send(msg)

    async def recv(self):
        return self.__pipe.recv()

    def join(self, timeout: float | None = None):
        self.__pipe.close()
        super().join(timeout)


if __name__ == "__main__":
    pass
