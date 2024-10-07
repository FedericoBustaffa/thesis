import pickle
import threading

from loguru import logger
from test.support import interpreters

from ppga import Crossoverator, Evaluator, Mutator


def task(
    r_chann: interpreters.RecvChannel,
    s_chann: interpreters.SendChannel,
    crossoverator: Crossoverator,
    mutator: Mutator,
    evaluator: Evaluator,
):
    interp = interpreters.create()
    source = open("modules/subinterp_source.py").read()
    interp.run("from test.support import interpreters")
    interp.run(f"r_chann = interpreters.RecvChannel({r_chann.id})")
    interp.run(f"s_chann = interpreters.SendChannel({s_chann.id})")
    s_chann.send(pickle.dumps(crossoverator))
    s_chann.send(pickle.dumps(mutator))
    s_chann.send(pickle.dumps(evaluator))
    interp.run(source)
    interp.close()


class SubInterpWorker:
    def __init__(self, crossoverator, mutator, evaluator) -> None:
        self.__rchann, self.__schann = interpreters.create_channel()
        self.__thread = threading.Thread(
            target=task,
            args=[self.__rchann, self.__schann, crossoverator, mutator, evaluator],
        )

    def start(self) -> None:
        self.__thread.start()

    def send(self, msg) -> None:
        self.__schann.send(pickle.dumps(msg))

    def recv(self):
        return pickle.loads(self.__rchann.recv())

    def join(self) -> None:
        self.__thread.join()


if __name__ == "__main__":
    pass
