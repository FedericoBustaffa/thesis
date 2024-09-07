import math

import multiprocessing as mp
from multiprocessing.connection import Connection

from pure import Genome


def generate(size: int, chromosome_length: int, gen_func, *args) -> list[Genome]:
    chromosomes = []
    for _ in range(size):
        c = gen_func(chromosome_length, *args)
        while c in chromosomes:
            c = gen_func(chromosome_length, *args)
        chromosomes.append(c)

    return [Genome(c) for c in chromosomes]


def evaluate(individuals: list[Genome], connection: Connection, fitness_func, *args):
    # print(f"{mp.current_process().name}: {len(individuals)}")
    for i in individuals:
        i.fitness = fitness_func(i.chromosome, *args)
        # print(i)

    connection.send(individuals)


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
