import multiprocessing.connection as conn
import time

from chromosome import Chromosome


def work(self, pipe: conn.Connection):
    timings = {"crossover": 0.0, "mutation": 0.0, "evaluation": 0.0}
    couples = pipe.recv()
    offsprings = [Chromosome([]) for _ in range(len(couples) * 2)]
    while couples != None:
        for i in range(len(couples)):
            father, mother = couples[i]
            start = time.perf_counter()
            o1, o2 = self.crossover_func(father, mother)
            timings["crossover"] += time.perf_counter() - start
            # print(f"{mp.current_process().name} offsprings: {o1}, {o2}")

            start = time.perf_counter()
            o1 = self.mutation_func(o1)
            o2 = self.mutation_func(o2)
            timings["mutation"] += time.perf_counter() - start
            # print(f"{mp.current_process().name} mutated offsprings: {o1}, {o2}")

            start = time.perf_counter()
            offsprings[i * 2] = Chromosome(o1, self.fitness_func(o1))
            offsprings[i * 2 + 1] = Chromosome(o2, self.fitness_func(o2))
            timings["evaluation"] += time.perf_counter() - start

        pipe.send(offsprings)
        couples = pipe.recv()

    pipe.send(timings)
    pipe.close()
