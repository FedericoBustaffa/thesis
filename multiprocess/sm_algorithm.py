import multiprocessing as mp
import multiprocessing.shared_memory as sm
import random

import numpy as np


class SharedMemoryGeneticAlgorithm:
    def __init__(
        self,
        population_size,
        gen_func,
        fitness_func,
        selection_func,
        crossover_func,
        mutation_func,
        mutation_rate,
        replace_func,
        num_of_workers: int = mp.cpu_count(),
    ) -> None:

        self.population_size = population_size
        self.gen_func = gen_func
        self.fitness_func = fitness_func
        self.selection_func = selection_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.mutation_rate = mutation_rate
        self.replace_func = replace_func

        self.workers = [
            mp.Process(target=self.parallel_work, args=[i, num_of_workers])
            for i in range(num_of_workers)
        ]

        for w in self.workers:
            w.start()
            w.join()

        # statistics
        self.average_fitness = []
        self.best_fitness = []
        self.biodiversity = []
        self.timings = {
            "generation": 0.0,
            "evaluation": 0.0,
            "selection": 0.0,
            "crossover": 0.0,
            "mutation": 0.0,
            "replacement": 0.0,
        }

    def generate(self):
        self.population = []
        self.scores = []

        for _ in range(self.population_size):
            chromosome = self.gen_func()
            while chromosome in self.population:
                chromosome = self.gen_func()

            self.population.append(chromosome)
            self.scores.append(self.fitness_func(chromosome))

        self.population = np.array(self.population)
        self.scores = np.array(self.scores)

        self.population_memory = sm.SharedMemory(
            name="population_memory", create=True, size=self.population.nbytes
        )

        self.scores_memory = sm.SharedMemory(
            name="scores_memory", create=True, size=self.scores.nbytes
        )

        self.couples_memory = sm.SharedMemory(
            name="couples_memory", create=True, size=self.scores.nbytes // 2
        )

    def selection(self):
        self.selected = self.selection_func(self.scores)

    def mating(self):
        couples = []
        for _ in range(len(self.selected) // 2):
            father, mother = random.choices(self.selected, k=2)
            # controllo father != mother ?

            couples.append([father, mother])

        couples = np.array(couples)

    def parallel_work(self, index: int, num_of_workers: int, couples_memory_name: str):
        couples_memory = sm.SharedMemory(name=couples_memory_name)
        couples = np.ndarray(buffer=couples_memory.buf)

    def run(self, max_generations: int) -> None:
        # initial population gen
        self.generate()
        print(f"generated")
        for i, s in zip(self.population, self.scores):
            print(f"{i}: {s}")

        for g in range(max_generations):
            # --- selection ---
            self.selection()
            print(f"selected")
            for i in self.selected:
                print(f"{self.population[i]}: {self.scores[i]}")

            # --- mating ---

            # should be in shared memory
            self.mating()

    def get(self):
        pass
