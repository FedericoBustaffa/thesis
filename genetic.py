import time

import numpy as np


class GeneticAlgorithm:

    def __init__(
        self,
        population_size: int,
        generation_func,
        fitness_func,
        selection_func,
        crossover_func,
        mutation_func,
        mutation_rate,
        replace_func,
    ):
        self.population_size = population_size
        self.generation_func = generation_func
        self.fitness_func = fitness_func
        self.selection_func = selection_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.mutation_rate = mutation_rate
        self.replace_func = replace_func

        # memory allocation
        self.population = np.empty(population_size, dtype=np.ndarray)
        self.scores = np.zeros(population_size)
        self.offsprings = np.empty(population_size // 2, dtype=np.ndarray)
        self.offsprings_scores = np.zeros(population_size // 2)

        # statistics
        self.average_fitness = []
        self.best_fitness = []
        self.biodiversities = []
        self.timings = {
            "generation": 0.0,
            "evaluation": 0.0,
            "selection": 0.0,
            "crossover": 0.0,
            "inner_crossover": 0.0,
            "mutation": 0.0,
            "replacement": 0.0,
        }

    def generation(self) -> None:
        for i in range(self.population_size):
            c = self.generation_func()
            while any(np.array_equal(arr, c) for arr in self.population):
                c = self.generation_func()

            self.population[i] = c

        self.scores = list(map(self.fitness_func, self.population))
        self.population, self.scores = (
            list(l)
            for l in zip(
                *sorted(
                    zip(self.population, self.scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        )

    def evaluation(self) -> None:
        self.offsprings_scores = list(map(self.fitness_func, self.offsprings))
        self.offsprings, self.offsprings_scores = (
            list(l)
            for l in zip(
                *sorted(
                    zip(self.offsprings, self.offsprings_scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        )

    def selection(self) -> None:
        self.selected = self.selection_func(self.population, self.scores)

    def crossover(self) -> None:
        indices = [i for i in range(len(self.selected))]
        for i in range(0, len(self.selected), 2):
            father_idx, mother_idx = np.random.choice(indices, 2)
            while father_idx == -1 or mother_idx == -1:
                father_idx, mother_idx = np.random.choice(indices, 2)

            father = self.population[self.selected[father_idx]]
            mother = self.population[self.selected[mother_idx]]
            start = time.perf_counter()
            offspring1, offspring2 = self.crossover_func(father, mother)
            end = time.perf_counter()
            self.timings["inner_crossover"] += end - start

            try:
                self.offsprings[i] = offspring1
                self.offsprings[i + 1] = offspring2
            except IndexError:
                pass

            self.selected[father_idx] = -1
            self.selected[mother_idx] = -1

    def mutation(self) -> None:
        for offspring in self.offsprings:
            if np.random.random() < self.mutation_rate:
                offspring = self.mutation_func(offspring)

    def replace(self) -> None:
        self.population, self.scores = self.replace_func(
            self.population, self.scores, self.offsprings, self.offsprings_scores
        )

    def get_best(self) -> tuple:
        return self.best

    def get_average_fitness(self) -> np.ndarray:
        return np.array(self.average_fitness)

    def get_best_fitness(self) -> np.ndarray:
        return np.array(self.best_fitness)

    def get_biodiversity(self) -> np.ndarray:
        return np.array(self.biodiversities)

    def get_timings(self) -> dict:
        return self.timings

    def run(self, generations):

        start = time.perf_counter()
        self.generation()
        end = time.perf_counter()
        self.timings["generation"] += end - start

        start = time.perf_counter()
        self.best = self.population[0], self.scores[0]
        end = time.perf_counter()
        self.timings["evaluation"] = end - start

        print(f"first best score: {self.best[1]}")

        for g in range(generations):

            start = time.perf_counter()
            self.selection()
            end = time.perf_counter()
            self.timings["selection"] += end - start

            start = time.perf_counter()
            self.crossover()
            end = time.perf_counter()
            self.timings["crossover"] += end - start

            start = time.perf_counter()
            self.mutation()
            end = time.perf_counter()
            self.timings["mutation"] += end - start

            start = time.perf_counter()
            self.evaluation()
            end = time.perf_counter()
            self.timings["evaluation"] += end - start

            start = time.perf_counter()
            self.replace()
            end = time.perf_counter()
            self.timings["replacement"] += end - start

            if self.best[1] < self.scores[0]:
                self.best = self.population[0], self.scores[0]

            self.average_fitness.append(np.mean(self.scores))
            self.biodiversities.append(
                len(set([tuple(i) for i in self.population]))
                / len(self.population)
                * 100.0
            )
            self.best_fitness.append(self.best[1])

            # convergence check
            if self.best[1] <= self.average_fitness[-1]:
                print(f"stop at generation {g}")
                break
