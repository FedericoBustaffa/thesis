import random
from functools import partial
import numpy as np


class GeneticAlgorithm:
    def __init__(
        self,
        population_size,
        chromosome_length,
        fitness_func,
        *fitness_func_args,
    ):
        self.population = np.array()
        self.population_fitness = []

        self.selected = []
        self.offsprings = []

        self.fitness_func = fitness_func
        self.fitness_func_args = fitness_func_args

    def generation(self, size: int, gen_func, *args) -> None:
        for _ in range(size):
            c = gen_func(*args)
            while c in self.population:
                c = gen_func(*args)
            self.population.append(c)

        self.population_fitness = np.array(
            [self.fitness_func(c, *self.fitness_func_args) for c in self.population]
        )

    def selection(self, selection_func) -> None:
        self.selected = selection_func(self.population)

    def crossover(self, crossover_func) -> None:
        self.offsprings.clear()
        while len(self.selected) > 0:
            father, mother = random.choices(self.selected, k=2)
            # while father.chromosome == mother.chromosome:
            #     print("crossover conflict")
            #     mother = random.choice(selected)
            offspring1, offspring2 = crossover_func(father, mother)

            self.offsprings.extend([offspring1, offspring2])
            self.selected.remove(father)
            try:
                self.selected.remove(mother)
            except ValueError:
                pass

    def evaluation(self, fitness, *args) -> None:
        self.offsprings_fitness = np.sort(
            list(map(partial(fitness, *args), self.offsprings))
        )

    def mutation(self, mutation_func, rate: float) -> None:
        for offspring in self.offsprings:
            if random.random() < rate:
                offspring = mutation_func(offspring)

    def replace(self, replace_func) -> None:
        self.population = replace_func(self.population, self.offsprings)

    def get_best(self) -> tuple:
        return self.population[0], self.population_fitness[0]

    def average_fitness(self) -> float:
        return sum([i.fitness for i in self.population]) / len(self.population)

    def biodiversity(self) -> float:
        return len(set(self.population)) / len(self.population) * 100.0
