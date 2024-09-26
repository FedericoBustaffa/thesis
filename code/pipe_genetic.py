import math
import multiprocessing as mp
import multiprocessing.connection as conn
import random
import time

import numpy as np


class PipeGeneticAlgorithm:

    def __init__(
        self,
        population_size: int,
        chromosome_length: int,
        generation_func,
        fitness_func,
        selection_func,
        crossover_func,
        mutation_func,
        mutation_rate: float,
        replace_func,
        workers_num: int = mp.cpu_count(),
    ) -> None:

        # setup
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

        # workers
        self.pipes = [mp.Pipe() for _ in range(workers_num)]
        self.workers = [
            mp.Process(target=self.parallel_work, args=[self.pipes[i][1]])
            for i in range(workers_num)
        ]
        for i in range(workers_num):
            self.workers[i].start()

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
            "statistics": 0.0,
        }

    def generation(self) -> None:

        # generate a new population
        start = time.perf_counter()
        self.population = []
        for _ in range(self.population_size):
            chromosome = self.generation_func()
            while any(np.array_equal(chromosome, i) for i in self.population):
                chromosome = self.generation_func()

            self.population.append(chromosome)
        self.population = np.array(self.population)
        self.timings["generation"] += time.perf_counter() - start

        # evaluate the generated population
        start = time.perf_counter()
        self.scores = []
        for i in self.population:
            self.scores.append(self.fitness_func(i))
        self.scores = np.array(self.scores)
        self.timings["evaluation"] += time.perf_counter() - start

        # preallocate memory for faster crossover
        # offsprings shared memory
        self.offsprings = np.zeros(
            shape=(self.population_size // 2, self.chromosome_length), dtype=np.int64
        )
        self.offsprings_scores = np.zeros(len(self.offsprings))

    def selection(self) -> None:
        start = time.perf_counter()
        self.selected = self.selection_func(self.scores)
        self.timings["selection"] += time.perf_counter() - start

    def mating(self) -> None:
        start = time.perf_counter()
        indices = np.random.choice(
            self.selected, size=(len(self.selected) // 2, 2), replace=False
        )
        self.couples = self.population[indices]
        self.timings["mating"] += time.perf_counter() - start

    def parallel(self) -> None:
        start = time.perf_counter()

        couples_chunk = math.ceil(len(self.couples) / self.workers_num)
        for i in range(len(self.pipes)):
            self.pipes[i][0].send(
                self.couples[i * couples_chunk : i * couples_chunk + couples_chunk]
            )

        offsprings_chunk = couples_chunk * 2
        for i in range(len(self.pipes)):
            offsprings_part = self.pipes[i][0].recv()
            self.offsprings[
                i * offsprings_chunk : i * offsprings_chunk + offsprings_chunk
            ] = offsprings_part

            self.offsprings_scores[
                i * offsprings_chunk : i * offsprings_chunk + offsprings_chunk
            ] = self.pipes[i][0].recv()

        self.parallel_time += time.perf_counter() - start

    def parallel_work(self, pipe: conn.Connection) -> None:
        timings = {
            "crossover": 0.0,
            "mutation": 0.0,
            "evaluation": 0.0,
        }

        while True:

            couples = pipe.recv()
            if couples is None:
                break

            offsprings = np.zeros(
                shape=(len(couples) * 2, self.chromosome_length), dtype=np.int64
            )
            scores = np.zeros(len(offsprings))

            for i in range(len(couples)):
                # crossover
                start = time.perf_counter()
                offsprings[i * 2], offsprings[i * 2 + 1] = self.crossover_func(
                    couples[i, 0], couples[i, 1]
                )
                timings["crossover"] += time.perf_counter() - start

                # mutation
                start = time.perf_counter()
                if random.random() < self.mutation_rate:
                    offsprings[i * 2] = self.mutation_func(offsprings[i * 2])
                if random.random() < self.mutation_rate:
                    offsprings[i * 2 + 1] = self.mutation_func(offsprings[i * 2 + 1])
                timings["mutation"] += time.perf_counter() - start

                # evaluation
                start = time.perf_counter()
                scores[i * 2] = self.fitness_func(offsprings[i * 2])
                scores[i * 2 + 1] = self.fitness_func(offsprings[i * 2 + 1])
                timings["evaluation"] += time.perf_counter() - start

            pipe.send(offsprings)
            pipe.send(scores)

        pipe.send(timings)
        pipe.close()

    def replace(self) -> None:
        start = time.perf_counter()
        self.population, self.scores = self.replace_func(
            self.population, self.scores, self.offsprings, self.offsprings_scores
        )
        self.timings["replacement"] += time.perf_counter() - start

    def run(self, max_generations: int) -> None:

        self.generation()

        self.best = self.population[0]
        self.best_score = self.scores[0]
        # print(f"first best: {self.best_score}")

        self.parallel_time = 0.0
        for g in range(max_generations):
            # print(f"generation: {g}")

            self.selection()
            self.mating()
            self.parallel()
            self.replace()

            if self.best_score < self.scores[0]:
                self.best = self.population[0]
                self.best_score = self.scores[0]

            # statistics
            start = time.perf_counter()
            self.average_fitness.append(np.mean(self.scores))

            self.biodiversity.append(
                len(np.unique(self.population, axis=0)) / len(self.population) * 100.0
            )

            self.best_fitness.append(self.best_score)
            self.timings["statistics"] += time.perf_counter() - start

            # convergence check
            if self.best_score <= self.average_fitness[-1]:
                print(f"stop at generation {g+1}")
                print(f"best score: {self.best_score}")
                print(f"average fitness: {self.average_fitness[-1]}")
                break

        for i in range(self.workers_num):
            self.pipes[i][0].send(None)

            timings = self.pipes[i][0].recv()
            if self.timings["crossover"] < timings["crossover"]:
                self.timings["crossover"] = timings["crossover"]

            if self.timings["mutation"] < timings["mutation"]:
                self.timings["mutation"] = timings["mutation"]

            if self.timings["evaluation"] < timings["evaluation"]:
                self.timings["evaluation"] = timings["evaluation"]

            self.pipes[i][0].close()
            self.workers[i].join()

        genetic_parallel_time = (
            self.timings["crossover"]
            + self.timings["mutation"]
            + self.timings["evaluation"]
        )
        # print(f"parallel time: {self.parallel_time} seconds")
        # print(
        #     f"parallel sync time: {self.parallel_time - genetic_parallel_time} seconds"
        # )
