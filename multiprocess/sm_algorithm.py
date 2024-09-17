import multiprocessing as mp
import multiprocessing.shared_memory as sm
import multiprocessing.synchronize as sync
import random

import numpy as np


def share(buffer, mem_name):

    buffer = np.array(buffer)
    buffer_memory = sm.SharedMemory(name=mem_name, create=True, size=buffer.nbytes)

    shared_buffer = np.ndarray(
        shape=buffer.shape,
        dtype=buffer.dtype,
        buffer=buffer_memory.buf,
    )

    shared_buffer[:] = buffer[:]

    return buffer_memory, shared_buffer


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

        self.condition = mp.Condition()
        self.workers = [
            mp.Process(
                target=self.parallel_work, args=[i, num_of_workers, self.condition]
            )
            for i in range(num_of_workers)
        ]

        for w in self.workers:
            w.start()

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
        population = []
        scores = []

        for _ in range(self.population_size):
            chromosome = self.gen_func()
            while chromosome in population:
                chromosome = self.gen_func()

            population.append(chromosome)
            scores.append(self.fitness_func(chromosome))

        # creating a shared memory for the population and scores
        self.population_memory, self.population = share(population, "population_mem")
        self.scores_memory, self.scores = share(scores, "scores_mem")

    def selection(self):
        self.selected = self.selection_func(self.scores)
        couples_buffer = [[-1, -1] for _ in range(len(self.selected) // 2)]
        self.couples_memory, self.couples = share(couples_buffer, "couples_mem")

    def mating(self):
        couples = []
        for _ in range(len(self.selected) // 2):
            father, mother = random.choices(self.selected, k=2)
            # controllo father != mother ?
            couples.append([father, mother])

        self.couples[:] = np.array(couples)[:]

    def parallel_work(self, index: int, num_of_workers: int, condition: sync.Condition):

        with condition:
            condition.wait()

        couples_memory = sm.SharedMemory(name="couples_mem")
        couples = np.ndarray(
            shape=(len(self.population_size) // 2, 2),
            dtype=np.int64,
            buffer=couples_memory.buf,
        )

        with condition:
            condition.wait()

        for c in couples:
            print(f"{mp.current_process().name}: {c}")

    def replace(self):
        # self.replace_func(self.population, self.scores, self.offsprings, self.offsprings_scores)
        pass

    def run(self, max_generations: int) -> None:
        # initial population gen
        self.generate()
        print(f"generated")
        for i, s in zip(self.population, self.scores):
            print(f"{i}: {s}")

        for g in range(max_generations):
            print(f"generation: {g+1}")

            # --- selection ---
            self.selection()

            with self.condition:
                self.condition.notify_all()

            print(f"selected")
            for i in self.selected:
                print(f"{self.population[i]}: {self.scores[i]}")

            # --- mating ---
            self.mating()

            # to delete (should be processes work)
            with self.condition:
                self.condition.notify_all()

            self.replace()

        for w in self.workers:
            w.join()

        self.population_memory.unlink()
        self.scores_memory.unlink()
        self.couples_memory.unlink()

    def get(self):
        pass
