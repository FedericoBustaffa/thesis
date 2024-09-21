import multiprocessing as mp
import multiprocessing.shared_memory as sm
import multiprocessing.sharedctypes as st
import multiprocessing.synchronize as sync
import random

import numpy as np
from parallel import parallel_work, share


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
        workers_num: int = mp.cpu_count(),
    ) -> None:

        self.population_size = population_size
        self.gen_func = gen_func
        self.fitness_func = fitness_func
        self.selection_func = selection_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.mutation_rate = mutation_rate
        self.replace_func = replace_func
        self.workers_num = workers_num

        # processes sync
        self.ready = mp.Event()
        self.ready_counter = mp.Value("i", workers_num)
        self.pipes = [mp.Pipe() for _ in range(workers_num)]

        self.workers = [
            mp.Process(
                target=parallel_work,
                args=[
                    self,
                    i,
                    workers_num,
                    self.pipes[i][1],
                    self.ready,
                    self.ready_counter,
                ],
            )
            for i in range(workers_num)
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

        # create a shared memory for the population and scores
        self.population_memory, self.population = share(population, "population_mem")
        self.scores_memory, self.scores = share(scores, "scores_mem")

        # create a shared memory for couples
        couples_buffer = [[-1, -1] for _ in range(len(self.population) // 4)]
        self.couples_memory, self.couples = share(couples_buffer, "couples_mem")

        for p in self.pipes:
            p[0].send((self.couples.shape, self.couples.dtype))

    def selection(self):
        self.selected = self.selection_func(self.scores)

    def mating(self):
        couples = []
        for _ in range(len(self.selected) // 2):
            father, mother = random.choices(self.selected, k=2)
            # controllo father != mother ?
            couples.append([father, mother])
            self.selected.remove(father)
            try:
                self.selected.remove(mother)
            except:
                pass

        self.couples[:] = np.array(couples)[:]

    def replace(self):
        # self.replace_func(
        #     self.population, self.scores, self.offsprings, self.offsprings_scores
        # )
        pass

    def run(self, max_generations: int) -> None:

        self.generate()

        self.selection()
        self.mating()

        self.ready.set()
        print("Ok")

        with self.ready_counter:
            while self.ready_counter.value != 0:
                self.ready.clear()
                self.ready.wait()

        print("main takes control")

        # for g in range(max_generations):
        #     print(f"generation: {g+1}")

        #     # --- selection ---
        #     self.selection()

        #     with self.condition:
        #         self.condition.notify_all()

        #     print(f"selected")
        #     for i in self.selected:
        #         print(f"{self.population[i]}: {self.scores[i]}")

        #     # --- mating ---
        #     self.mating()

        #     # to delete (should be processes work)
        #     with self.condition:
        #         self.main_ready.value = 1
        #         self.condition.notify_all()

        #     with self.condition:
        #         self.condition.wait_for(
        #             lambda: self.process_ready_counter.value == len(self.workers)
        #         )

        #     self.replace()

        for i in range(len(self.workers)):
            self.pipes[i][0].close()
            self.workers[i].join()

        for w in self.workers:
            w.close()

        del self.population
        del self.scores
        del self.couples

        self.population_memory.close()
        self.scores_memory.close()
        self.couples_memory.close()

        self.population_memory.unlink()
        self.scores_memory.unlink()
        self.couples_memory.unlink()

    def get(self):
        pass
