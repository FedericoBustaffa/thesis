import multiprocessing as mp
import multiprocessing.connection as conn
import multiprocessing.shared_memory as sm
import multiprocessing.sharedctypes as st
import multiprocessing.synchronize as sync
import random
import time

import numpy as np


def share(buffer, mem_name):

    buffer = np.array(buffer)
    buffer_memory = sm.SharedMemory(name=mem_name, create=True, size=buffer.nbytes)

    shared_buffer = np.ndarray(
        shape=buffer.shape,
        dtype=buffer.dtype,
        buffer=buffer_memory.buf,
    )
    np.copyto(shared_buffer, buffer)

    return buffer_memory, shared_buffer


class SharedMemoryGeneticAlgorithm:

    def __init__(
        self,
        population_size: int,
        chromosome_length: int,
        generation_func,
        fitness_func,
        selection_func,
        crossover_func,
        mutation_func,
        mutation_rate,
        replace_func,
        workers_num: int = mp.cpu_count(),
    ) -> None:

        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.generation_func = generation_func
        self.fitness_func = fitness_func
        self.selection_func = selection_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.mutation_rate = mutation_rate
        self.replace_func = replace_func
        self.workers_num = workers_num

        # synchronization primitives
        self.pipes = [mp.Pipe() for _ in range(workers_num)]
        self.main_ready = [mp.Event() for _ in range(workers_num)]
        self.workers_ready = [mp.Event() for _ in range(workers_num)]
        self.stops = [mp.Value("i", 0) for _ in range(workers_num)]

        # for shared memory
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
            "replacement": 0.0,
        }

    def generation(self):

        # generate a new population
        start = time.perf_counter()
        population = []
        for _ in range(self.population_size):
            chromosome = self.generation_func()
            while any(np.array_equal(chromosome, i) for i in population):
                chromosome = self.generation_func()

            population.append(chromosome)
        self.timings["generation"] += time.perf_counter() - start

        # sharing population
        self.population_memory, self.population = share(population, "population_mem")
        self.shapes.append(self.population.shape)
        self.dtypes.append(self.population.dtype)

        # evaluate the generated population
        start = time.perf_counter()
        scores = []
        for i in population:
            scores.append(self.fitness_func(i))
        self.timings["evaluation"] += time.perf_counter() - start

        # sharing scores
        self.scores_memory, self.scores = share(scores, "scores_mem")
        self.shapes.append(self.scores.shape)
        self.dtypes.append(self.scores.dtype)

        # preallocate memory for faster crossover
        # offsprings shared memory
        offsprings = [
            [0 for _ in range(self.chromosome_length)]
            for _ in range(self.population_size // 2)
        ]
        self.offsprings_memory, self.offsprings = share(offsprings, "offsprings_mem")
        self.shapes.append(self.offsprings.shape)
        self.dtypes.append(self.offsprings.dtype)

        # offsprings scores sharing
        offsprings_scores = [0.0 for _ in range(len(offsprings))]
        self.offsprings_scores_memory, self.offsprings_scores = share(
            offsprings_scores, "offsprings_scores_mem"
        )
        self.shapes.append(self.offsprings_scores.shape)
        self.dtypes.append(self.offsprings_scores.dtype)

        # couples indices shared memory
        couples = [[-1, -1] for _ in range(self.population_size // 4)]
        self.couples_memory, self.couples = share(couples, "couples_mem")
        self.shapes.append(self.couples.shape)
        self.dtypes.append(self.couples.dtype)

    def selection(self):
        start = time.perf_counter()
        self.selected = self.selection_func(self.scores)
        self.timings["selection"] += time.perf_counter() - start

    def mating(self):
        start = time.perf_counter()
        couples = np.random.choice(
            self.selected, size=(len(self.selected) // 2, 2), replace=False
        )
        self.couples[:] = couples[:]
        self.timings["mating"] += time.perf_counter() - start

    def replace(self):
        start = time.perf_counter()
        population, scores = self.replace_func(
            self.population, self.scores, self.offsprings, self.offsprings_scores
        )

        np.copyto(self.population, population)
        np.copyto(self.scores, scores)
        self.timings["replacement"] += time.perf_counter() - start

    def start_workers(self):

        self.workers = [
            mp.Process(
                target=self.parallel_work,
                args=[
                    i,
                    self.workers_num,
                    self.shapes,
                    self.dtypes,
                    self.pipes[i][1],
                    self.main_ready[i],
                    self.workers_ready[i],
                    self.stops[i],
                ],
            )
            for i in range(self.workers_num)
        ]

        for w in self.workers:
            w.start()

    def parallel_work(
        self,
        index: int,
        workers_num: int,
        shapes,
        dtypes,
        pipe: conn.Connection,
        main_ready: sync.Event,
        ready: sync.Event,
        stop: st.Synchronized,
    ):

        population_memory = sm.SharedMemory(name="population_mem")
        population = np.ndarray(
            shape=shapes[0],
            dtype=dtypes[0],
            buffer=population_memory.buf,
        )

        scores_memory = sm.SharedMemory(name="scores_mem")
        scores = np.ndarray(
            shape=shapes[1],
            dtype=dtypes[1],
            buffer=scores_memory.buf,
        )

        offsprings_memory = sm.SharedMemory(name="offsprings_mem")
        offsprings = np.ndarray(
            shape=shapes[2],
            dtype=dtypes[2],
            buffer=offsprings_memory.buf,
        )

        offsprings_scores_memory = sm.SharedMemory(name="offsprings_scores_mem")
        offsprings_scores = np.ndarray(
            shape=shapes[3],
            dtype=dtypes[3],
            buffer=offsprings_scores_memory.buf,
        )

        couples_memory = sm.SharedMemory(name="couples_mem")
        couples = np.ndarray(
            shape=shapes[4],
            dtype=dtypes[4],
            buffer=couples_memory.buf,
        )

        timings = {
            "crossover": 0.0,
            "mutation": 0.0,
            "evaluation": 0.0,
        }

        chunk_size = len(couples) // workers_num
        while True:
            main_ready.wait()
            main_ready.clear()

            with stop:
                if stop.value == 1:
                    break
                for i in range(index * chunk_size, index * chunk_size + chunk_size, 1):
                    # crossover
                    start = time.perf_counter()
                    offsprings[i * 2], offsprings[i * 2 + 1] = self.crossover_func(
                        population[couples[i, 0]], population[couples[i, 1]]
                    )
                    timings["crossover"] += time.perf_counter() - start

                    # mutation
                    start = time.perf_counter()
                    if random.random() < self.mutation_rate:
                        offsprings[i * 2] = self.mutation_func(offsprings[i * 2])
                    if random.random() < self.mutation_rate:
                        offsprings[i * 2 + 1] = self.mutation_func(
                            offsprings[i * 2 + 1]
                        )
                    timings["mutation"] += time.perf_counter() - start

                    # evaluation
                    start = time.perf_counter()
                    offsprings_scores[i * 2] = self.fitness_func(offsprings[i * 2])
                    offsprings_scores[i * 2 + 1] = self.fitness_func(
                        offsprings[i * 2 + 1]
                    )
                    timings["evaluation"] += time.perf_counter() - start

                ready.set()

        couples_memory.close()
        pipe.send(timings)
        pipe.close()

    def unlink(self):
        try:
            self.couples_memory.unlink()
            self.population_memory.unlink()
            self.scores_memory.unlink()
            self.offsprings_memory.unlink()
            self.offsprings_scores_memory.unlink()
        except:
            print("shared memory exception")

    def run(self, max_generations: int) -> None:
        self.generation()

        self.best = self.population[0]
        self.best_score = self.scores[0]
        print(f"first best: {self.best_score}")

        self.start_workers()

        genetic_time = time.perf_counter()
        parallel_time = 0.0
        for g in range(max_generations):
            # print(f"generation: {g+1}")

            self.selection()
            self.mating()

            start = time.perf_counter()
            for main_ready in self.main_ready:
                main_ready.set()

            for worker_ready in self.workers_ready:
                worker_ready.wait()
                worker_ready.clear()
            parallel_time += time.perf_counter() - start

            self.replace()

            if self.best_score < self.scores[0]:
                self.best = self.population[0]
                self.best_score = self.scores[0]

            self.average_fitness.append(np.mean(self.scores))

            self.biodiversity.append(
                len(np.unique(self.population, axis=0)) / len(self.population) * 100.0
            )

            self.best_fitness.append(self.best_score)

            # convergence check
            # if self.best_score <= self.average_fitness[-1]:
            #     print(f"stop at generation {g+1}")
            #     print(f"best score: {self.best_score}")
            #     print(f"average fitness: {self.average_fitness[-1]}")
            #     break

        print(f"genetic time: {time.perf_counter() - genetic_time} seconds")
        for i in range(self.workers_num):
            with self.stops[i]:
                self.stops[i].value = 1
            self.main_ready[i].set()
            timings = self.pipes[i][0].recv()

            if self.timings["crossover"] < timings["crossover"]:
                self.timings["crossover"] = timings["crossover"]

            if self.timings["mutation"] < timings["mutation"]:
                self.timings["mutation"] = timings["mutation"]

            if self.timings["evaluation"] < timings["evaluation"]:
                self.timings["evaluation"] = timings["evaluation"]

            self.pipes[i][0].close()
            self.workers[i].join()

        self.unlink()

        genetic_parallel_time = (
            self.timings["crossover"]
            + self.timings["mutation"]
            + self.timings["evaluation"]
        )
        print(f"parallel global time: {parallel_time}")
        print(f"parallel sync time: {parallel_time - genetic_parallel_time}")
