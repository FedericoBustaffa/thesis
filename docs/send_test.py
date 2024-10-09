import multiprocessing as mp
import multiprocessing.queues as mpq
import random
import sys
import time

import numpy as np
from tqdm import tqdm


class Fitness:
    def __init__(self, weights: tuple) -> None:
        self.weights = weights
        self.values = None

    @property
    def fitness(self) -> float:
        if self.values is None:
            return 0.0
        return sum([v * w for v, w in zip(self.values, self.weights)])

    def __eq__(self, other) -> bool:
        assert isinstance(other, Fitness)
        return self.fitness == other.fitness

    def __lt__(self, other) -> bool:
        assert isinstance(other, Fitness)
        return self.fitness < other.fitness

    def __gt__(self, other) -> bool:
        assert isinstance(other, Fitness)
        return self.fitness > other.fitness

    def __str__(self) -> str:
        return str(self.fitness)

    def __repr__(self) -> str:
        return str(self.fitness)


class Individual:
    def __init__(self, chromosome, fitness: Fitness) -> None:
        self.chromosome = chromosome
        self.fitness = fitness

    def __repr__(self) -> str:
        return f"{self.chromosome}: {self.fitness.fitness}"

    def __eq__(self, other) -> bool:
        assert isinstance(other, Individual)
        return self.chromosome == other.chromosome

    def __lt__(self, other) -> bool:
        assert isinstance(other, Individual)
        return self.fitness < other.fitness

    def __gt__(self, other) -> bool:
        assert isinstance(other, Individual)
        return self.fitness > other.fitness


def receive(recv_queue: mpq.Queue):
    while True:
        buffer = recv_queue.get()
        if buffer is None:
            break


def main():
    recv_queue = mp.Queue()
    receiver = mp.Process(target=receive, args=[recv_queue])
    receiver.start()

    population = [
        Individual(
            [random.randint(0, 200) for _ in range(200)],
            Fitness((1.0,)),
        )
        for _ in range(10000)
    ]

    print(f"individual size: {sys.getsizeof(population[0])} bytes")

    timings = []
    tot_start = time.perf_counter()
    for i in tqdm(range(len(population)), ncols=80):
        start = time.perf_counter()
        recv_queue.put(population[i])
        end = time.perf_counter()
        timings.append((end - start) * 1000000.0)
    tot_time = time.perf_counter() - tot_start

    recv_queue.put(None)

    recv_queue.close()
    receiver.join()

    print(f"mean send time: {np.mean(timings)} microseconds")
    print(f"total send time: {tot_time * 1000.0} microseconds")


if __name__ == "__main__":
    main()
