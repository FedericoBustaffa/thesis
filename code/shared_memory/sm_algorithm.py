import multiprocessing as mp
import multiprocessing.shared_memory as sm
import random

import numpy as np
from parallel import parallel_work, share


class SharedMemoryGeneticAlgorithm:

    share = share
    worker_task = parallel_work

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
        self.pipes = [mp.Pipe() for _ in range(workers_num)]
        self.main_ready = [mp.Event() for _ in range(workers_num)]
        self.workers_ready = [mp.Event() for _ in range(workers_num)]
        self.stops = [mp.Value("i", 0) for _ in range(workers_num)]

        self.workers = [
            mp.Process(
                target=self.worker_task,
                args=[
                    i,
                    workers_num,
                    self.pipes[i][1],
                    self.main_ready[i],
                    self.workers_ready[i],
                    self.stops[i],
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
        self.population_memory, self.population = self.share(
            population, "population_mem"
        )
        self.scores_memory, self.scores = self.share(scores, "scores_mem")

        # create a shared memory for couples
        couples_buffer = [[-1, -1] for _ in range(len(self.population) // 4)]
        self.couples_memory, self.couples = self.share(couples_buffer, "couples_mem")

        for p in self.pipes:
            p[0].send((self.couples.shape, self.couples.dtype))
            p[0].send((self.population.shape, self.population.dtype))
            p[0].send((self.scores.shape, self.scores.dtype))

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

        for g in range(max_generations):
            print(f"generation: {g+1}")

            self.selection()
            self.mating()

            for main_ready in self.main_ready:
                main_ready.set()

            for worker_ready in self.workers_ready:
                worker_ready.wait()
                worker_ready.clear()

        for i in range(len(self.workers)):
            with self.stops[i]:
                self.stops[i].value = 1
            self.main_ready[i].set()
            self.pipes[i][0].close()
            self.workers[i].join()

        try:
            self.population_memory.unlink()
            self.scores_memory.unlink()
            self.couples_memory.unlink()
        except:
            print("shared memory exception")

    def get(self):
        pass
