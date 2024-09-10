import math

import multiprocessing as mp
from multiprocessing.connection import Connection

from genetic import Genome


def pipe_evaluate(connection: Connection, fitness_func, *args):
    individuals = connection.recv()
    while individuals != None:
        for i in individuals:
            i.fitness = fitness_func(i.chromosome, *args)

        connection.send(individuals)
        individuals = connection.recv()


class PipeEvaluator:
    def __init__(self, fitness_func, *args, cores: int = 0) -> None:
        self.cores = cores if cores != 0 else mp.cpu_count()
        self.pipes = [mp.Pipe() for _ in range(self.cores)]
        self.workers = [
            mp.Process(
                target=pipe_evaluate,
                args=[self.pipes[i][1], fitness_func, *args],
            )
            for i in range(self.cores)
        ]
        for w in self.workers:
            w.start()

    def evaluate(self, individuals: list[Genome]) -> None:
        portion = math.ceil(len(individuals) / self.cores)
        for i in range(self.cores):
            first = i * portion
            last = (
                first + portion
                if first + portion <= len(individuals)
                else len(individuals)
            )

            partial = individuals[first:last]
            self.pipes[i][0].send(partial)

        individuals.clear()
        for i in range(len(self.workers)):
            individuals.extend(self.pipes[i][0].recv())

        individuals.sort(key=lambda x: x.fitness, reverse=True)

    def shutdown(self) -> None:
        for i in range(self.cores):
            self.pipes[i][0].send(None)
            self.workers[i].join()
