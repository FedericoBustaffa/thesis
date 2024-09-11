import random
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

    def generation(self) -> None:
        self.population = []
        self.scores = np.zeros(self.population_size)

        for _ in range(self.population_size):
            c = self.generation_func()
            while any(np.array_equal(arr, c) for arr in self.population):
                c = self.generation_func()
            self.population.append(c)

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
        self.offsprings = []
        for i in range(len(self.selected) // 2):
            father_idx, mother_idx = random.choices(self.selected, k=2)
            father = self.population[father_idx]
            mother = self.population[mother_idx]
            # while father.chromosome == mother.chromosome:
            #     print("crossover conflict")
            #     mother = random.choice(selected)
            offspring1, offspring2 = self.crossover_func(father, mother)

            self.offsprings.extend([offspring1, offspring2])
            self.selected.remove(father_idx)
            try:
                self.selected.remove(mother_idx)
            except ValueError:
                pass

        self.offsprings_scores = np.zeros(len(self.offsprings))

    def mutation(self) -> None:
        for offspring in self.offsprings:
            if random.random() < self.mutation_rate:
                offspring = self.mutation_func(offspring)

    def replace(self) -> None:
        self.population, self.scores = self.replace_func(
            self.population, self.scores, self.offsprings, self.offsprings_scores
        )

    def get_best(self) -> tuple:
        return self.best

    def average_fitness(self) -> float:
        return self.scores.mean()

    def biodiversity(self) -> float:
        return len(set(self.population)) / len(self.population) * 100.0

    def run(self, generations):

        self.generation()
        self.best = self.population[0], self.scores[0]
        print(f"first best score: {self.best[1]}")

        for g in range(generations):
            self.selection()
            self.crossover()
            self.mutation()
            self.evaluation()
            self.replace()

            if self.best[1] < self.scores[0]:
                self.best = self.population[0], self.scores[0]
