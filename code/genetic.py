import random
import time

import numpy as np


class GeneticAlgorithm:

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
    ):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.generation_func = generation_func
        self.fitness_func = fitness_func
        self.selection_func = selection_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.mutation_rate = mutation_rate
        self.replace_func = replace_func

        # statistics
        self.average_fitness = []
        self.best_fitness = []
        self.biodiversity = []
        self.timings = {
            "generation": 0.0,
            "evaluation": 0.0,
            "selection": 0.0,
            "mating": 0.0,
            "crossover": 0.0,
            "mutation": 0.0,
            "replacement": 0.0,
        }

    def generation(self) -> None:

        self.population = []
        self.scores = np.zeros(self.population_size)

        start = time.perf_counter()
        for _ in range(self.population_size):
            chromosome = self.generation_func()
            while any(np.array_equal(chromosome, i) for i in self.population):
                chromosome = self.generation_func()

            self.population.append(chromosome)
        self.population = np.array(self.population)
        self.timings["generation"] += time.perf_counter() - start

        start = time.perf_counter()
        for i in range(self.population_size):
            self.scores[i] = self.fitness_func(self.population[i])
        self.timings["evaluation"] += time.perf_counter() - start

        # preallocate memory for faster crossover
        chromosome_length = len(self.population[0])
        self.offsprings = np.array(
            [
                np.zeros(chromosome_length, dtype=np.int64)
                for _ in range(self.population_size // 2)
            ]
        )
        self.offsprings_scores = np.zeros(len(self.offsprings))

    def selection(self) -> None:
        start = time.perf_counter()
        self.selected = self.selection_func(self.scores)
        self.timings["selection"] += time.perf_counter() - start

    def mating(self) -> None:
        start = time.perf_counter()
        self.couples = np.random.choice(
            self.selected, size=(len(self.selected) // 2, 2), replace=False
        )
        self.timings["mating"] += time.perf_counter() - start

    def crossover(self) -> None:
        start = time.perf_counter()
        for i in range(len(self.couples)):
            offspring1, offspring2 = self.crossover_func(
                self.population[self.couples[i, 0]],
                self.population[self.couples[i, 1]],
            )
            self.offsprings[i * 2] = offspring1
            self.offsprings[i * 2 + 1] = offspring2
        self.timings["crossover"] += time.perf_counter() - start

    def mutation(self) -> None:
        start = time.perf_counter()
        for offspring in self.offsprings:
            if random.random() < self.mutation_rate:
                offspring = self.mutation_func(offspring)
        self.timings["mutation"] += time.perf_counter() - start

    def evaluation(self) -> None:
        start = time.perf_counter()
        for i in range(len(self.offsprings)):
            self.offsprings_scores[i] = self.fitness_func(self.offsprings[i])
        self.timings["evaluation"] += time.perf_counter() - start

    def replace(self) -> None:
        start = time.perf_counter()
        self.population, self.scores = self.replace_func(
            self.population, self.scores, self.offsprings, self.offsprings_scores
        )
        self.timings["replacement"] += time.perf_counter() - start

    def run(self, generations):

        self.generation()

        self.best = self.population[0]
        self.best_score = self.scores[0]
        print(f"first best score: {self.best_score}")

        for g in range(generations):
            print(f"generation: {g+1}")

            self.selection()
            self.mating()
            self.crossover()
            self.mutation()
            self.evaluation()
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
            #     break
