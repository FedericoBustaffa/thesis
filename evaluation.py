import math

import multiprocessing as mp
from multiprocessing.connection import Connection

from pure import Genome


def evaluate(connection: Connection, fitness_func, *args):
    individuals = connection.recv()
    if individuals == None:
        return

    for i in individuals:
        i.fitness = fitness_func(i.chromosome, *args)
        print(i)

    connection.send(individuals)


class Evaluator:
    def __init__(self, fitness_func, *args, cores: int = 0) -> None:
        self.cores = cores if cores != 0 else mp.cpu_count()
        self.pipes = [mp.Pipe() for _ in range(self.cores)]
        self.workers = [
            mp.Process(
                target=evaluate,
                args=[self.pipes[i][1], fitness_func, *args],
            )
            for i in range(self.cores)
        ]
        for w in self.workers:
            w.start()

    def evaluate(self, individuals: list[Genome]) -> list[Genome]:
        portion = math.ceil(len(individuals) / self.cores)
        for i in range(self.cores):
            first = i * portion
            last = first + portion
            if last > len(individuals):
                last = len(individuals)

            partial = individuals[first:last]
            self.pipes[i][0].send(partial)

        individuals.clear()
        for i in range(len(self.workers)):
            self.workers[i].join()
            individuals.extend(self.pipes[i][0].recv())

        return sorted(individuals, key=lambda x: x.fitness, reverse=True)

    def shutdown(self) -> None:
        for i in range(self.cores):
            self.pipes[i][0].send(None)
            self.workers[i].join()


def evaluation(individuals: list[Genome], fitness_func, *args) -> list[Genome]:
    cores = mp.cpu_count()
    portion = math.ceil(len(individuals) / cores)

    pipes = [mp.Pipe(duplex=False) for _ in range(cores)]
    workers = []
    for i in range(cores):
        first = i * portion
        last = first + portion
        if last > len(individuals):
            last = len(individuals)

        partial = individuals[first:last]
        workers.append(
            mp.Process(
                target=evaluate, args=[partial, pipes[i][1], fitness_func, *args]
            )
        )
        workers[i].start()

    individuals.clear()
    for i in range(len(workers)):
        workers[i].join()
        individuals.extend(pipes[i][0].recv())

    return sorted(individuals, key=lambda x: x.fitness, reverse=True)
