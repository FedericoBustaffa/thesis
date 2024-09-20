import time

from chromosome import Chromosome


def generate(self):
    start = time.perf_counter()
    chromosomes = []
    for _ in range(self.population_size):
        values = self.gen_func()
        while values in chromosomes:
            values = self.gen_func()

        chromosomes.append(values)

    self.population = [
        Chromosome(values, self.fitness_func(values)) for values in chromosomes
    ]

    self.timings["generation"] += time.perf_counter() - start
