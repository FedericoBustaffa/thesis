import random
import time

import numpy as np
from parallel import share


def generate(self):
    population = []
    scores = []

    start = time.perf_counter()
    for _ in range(self.population_size):
        chromosome = self.gen_func()
        # while np.isin(chromosome, population):
        #     chromosome = self.gen_func()

        population.append(chromosome)
        scores.append(self.fitness_func(chromosome))

    # create a shared memory for couples
    couples_buffer = [[-1, -1] for _ in range(self.population_size // 4)]
    self.couples_memory, self.couples = share(couples_buffer, "couples_mem")
    self.shapes.append(self.couples.shape)
    self.dtypes.append(self.couples.dtype)

    # create a shared memory for the population and their scores
    self.population_memory, self.population = share(population, "population_mem")
    self.shapes.append(self.population.shape)
    self.dtypes.append(self.population.dtype)

    self.scores_memory, self.scores = share(scores, "scores_mem")
    self.shapes.append(self.scores.shape)
    self.dtypes.append(self.scores.dtype)

    # create a shared memory for offsprings and their scores
    offsprings = [
        [0 for _ in range(len(self.population[0]))]
        for _ in range(self.population_size // 2)
    ]
    self.offsprings_memory, self.offsprings = share(offsprings, "offsprings_mem")
    self.shapes.append(self.offsprings.shape)
    self.dtypes.append(self.offsprings.dtype)

    offsprings_scores = [0.0 for i in range(self.population_size // 2)]
    self.offsprings_scores_memory, self.offsprings_scores = share(
        offsprings_scores, "offsprings_scores_mem"
    )
    self.shapes.append(self.offsprings_scores.shape)
    self.dtypes.append(self.offsprings_scores.dtype)
    self.timings["generation"] += time.perf_counter() - start


def select(self):
    start = time.perf_counter()
    self.selected = self.selection_func(self.scores)
    self.timings["selection"] += time.perf_counter() - start


def mating(self):
    couples = []

    start = time.perf_counter()
    for _ in range(len(self.selected) // 2):
        father, mother = random.choices(self.selected, k=2)
        # controllo father != mother ?
        couples.append([father, mother])
        self.selected.remove(father)
        try:
            self.selected.remove(mother)
        except:
            pass
    self.timings["mating"] += time.perf_counter() - start

    self.couples[:] = np.array(couples)[:]


def replace(self):
    start = time.perf_counter()
    population, scores = self.replace_func(
        self.population, self.scores, self.offsprings, self.offsprings_scores
    )

    np.copyto(self.population, population)
    np.copyto(self.scores, scores)
    self.timings["replacement"] += time.perf_counter() - start


def unlink(self):
    try:
        self.couples_memory.unlink()
        self.population_memory.unlink()
        self.scores_memory.unlink()
        self.offsprings_memory.unlink()
        self.offsprings_scores_memory.unlink()
    except:
        print("shared memory exception")
