import multiprocessing as mp
import multiprocessing.shared_memory as sm
import time

import numpy as np

from genetic import (
    generate,
    mating,
    replace,
    select,
    unlink,
)
from parallel import parallel_work, start_workers


class SharedMemoryGeneticAlgorithm:

    # genetic functions and operators
    generate = generate
    select = select
    mating = mating
    replace = replace
    unlink = unlink

    # parallel methods
    start_workers = start_workers
    parallel_work = parallel_work

    def __init__(
        self,
        population_size: int,
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
        self.main_ready = [mp.Event() for _ in range(workers_num)]
        self.workers_ready = [mp.Event() for _ in range(workers_num)]
        self.stops = [mp.Value("i", 0) for _ in range(workers_num)]

        self.shapes = []
        self.dtypes = []

        # statistics
        self.average_fitness = []
        self.best_fitness = []
        self.biodiversity = []
        self.timings = {
            "generation": 0.0,
            "selection": 0.0,
            "mating": 0.0,
            "crossover": 0.0,
            "mutation": 0.0,
            "evaluation": 0.0,
            "parallel": 0.0,
            "replacement": 0.0,
            "stuff": 0.0,
        }

    def run(self, max_generations: int) -> None:

        self.generate()

        self.best = np.zeros(len(self.population[0]))
        np.copyto(self.best, self.population[0])
        self.best_score = self.scores[0]
        print(f"first best: {self.best_score}")

        self.start_workers()

        genetic_time = time.perf_counter()
        for g in range(max_generations):
            print(f"generation: {g+1}")

            self.select()
            self.mating()

            start = time.perf_counter()
            for main_ready in self.main_ready:
                main_ready.set()

            for worker_ready in self.workers_ready:
                worker_ready.wait()
                worker_ready.clear()
            self.timings["parallel"] += time.perf_counter() - start

            self.replace()

            start = time.perf_counter()
            if self.best_score < self.scores[0]:
                np.copyto(self.best, self.population[0])
                self.best_score = self.scores[0]

            self.average_fitness.append(self.scores.mean())
            self.best_fitness.append(self.best_score)

            # self.biodiversity.append(
            #     len(set(tuple(i) for i in self.population))
            #     / len(self.population)
            #     * 100.0
            # )
            self.timings["stuff"] += time.perf_counter() - start

            # convergence check
            # if self.best_score <= self.average_fitness[-1]:
            #     print(f"stop at generation {g+1}")
            #     print(f"best score: {self.best_score}")
            #     print(f"average fitness: {self.average_fitness[-1]}")
            #     break

        print(f"genetic time: {time.perf_counter() - genetic_time} seconds")
        for i in range(len(self.workers)):
            with self.stops[i]:
                self.stops[i].value = 1
            self.main_ready[i].set()
            self.workers[i].join()

        self.unlink()
