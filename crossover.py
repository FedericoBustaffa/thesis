import math
import random

import multiprocessing as mp
from multiprocessing.connection import Connection

from genetic import Genome


def pipe_crossover(connection: Connection, crossover_func):
    couples = connection.recv()
    offsprings = []
    while couples != None:
        for father, mother in couples:
            offspring1, offspring2 = crossover_func(father, mother)
            offsprings.extend([Genome(offspring1), Genome(offspring2)])

        connection.send(offsprings)
        offsprings.clear()
        couples = connection.recv()


class PipeCrossover:
    def __init__(self, crossover_func, cores: int = 0) -> None:
        self.cores = cores if cores != 0 else mp.cpu_count()
        self.pipes = [mp.Pipe() for _ in range(self.cores)]
        self.workers = [
            mp.Process(
                target=pipe_crossover,
                args=[self.pipes[i][1], crossover_func],
            )
            for i in range(self.cores)
        ]
        for w in self.workers:
            w.start()

    def crossover(self, selected: list[Genome]) -> list[Genome]:
        # creazione delle coppie
        couples = []
        for i in range(len(selected) // 2):
            father, mother = random.choices(selected, k=2)
            couples.append((father, mother))

        portion = math.ceil(len(couples) / self.cores)
        for i in range(self.cores):
            first = i * portion
            last = first + portion if first + portion <= len(couples) else len(couples)

            partial = couples[first:last]
            self.pipes[i][0].send(partial)

        offsprings = []
        for i in range(len(self.workers)):
            offsprings.extend(self.pipes[i][0].recv())

        return offsprings

    def shutdown(self):
        for i in range(len(self.workers)):
            self.pipes[i][0].send(None)
            self.workers[i].join()
