import random
import numpy as np


class GeneticAlgorithm:

    def __init__(self):
        self.population = []
        self.fitness_values = []

        self.offsprings = []
        self.offsprings_fitness = []

    def generation(self, size: int, gen_func, *args) -> None:
        self.population = []
        self.fitness_values = np.zeros(size)

        for _ in range(size):
            c = gen_func(*args)
            while any(np.array_equal(arr, c) for arr in self.population):
                c = gen_func(*args)
            self.population.append(c)

    def selection(self, selection_func) -> None:
        self.selected = selection_func(self.population, self.fitness_values)

    def crossover(self, crossover_func) -> None:
        for i in range(0, len(self.selected), 2):
            father, mother = random.choices(self.selected, k=2)
            father = self.population[father]
            mother = self.population[mother]
            # while father.chromosome == mother.chromosome:
            #     print("crossover conflict")
            #     mother = random.choice(selected)
            offspring1, offspring2 = crossover_func(father, mother)

            self.offsprings[i] = offspring1
            self.offsprings[i + 1] = offspring2
            self.selected.remove(father)
            self.selected.remove(mother)

    def evaluation(self, fitness, *args) -> None:
        for i in range(len(self.offsprings)):
            self.offsprings_fitness[i] = fitness(self.offsprings[i], *args)

    def mutation(self, mutation_func, rate: float) -> None:
        for offspring in self.offsprings:
            if random.random() < rate:
                offspring = mutation_func(offspring)

    def replace(self, replace_func) -> None:
        self.population = replace_func(self.population, self.offsprings)

    def get_best(self) -> tuple:
        return self.population[0], self.fitness_values[0]

    def average_fitness(self) -> float:
        return sum([i.fitness for i in self.population]) / len(self.population)

    def biodiversity(self) -> float:
        return len(set(self.population)) / len(self.population) * 100.0
