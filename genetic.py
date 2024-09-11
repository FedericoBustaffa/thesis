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
        self.population = []
        self.fitness_scores = []

    def generation(self, size: int, gen_func, *args) -> None:
        self.population = []
        self.scores = np.zeros(size)

        for _ in range(size):
            c = gen_func(*args)
            while any(np.array_equal(arr, c) for arr in self.population):
                c = gen_func(*args)
            self.population.append(c)

    def evaluation(self, individuals) -> None:
        for i in range(len(individuals)):
            self.scores[i] = self.fitness_func(individuals[i])

    def selection(self, selection_func) -> None:
        self.selected = selection_func(self.population, self.fitness_scores)

    def crossover(self) -> None:
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

    def mutation(self) -> None:
        for offspring in self.offsprings:
            if random.random() < self.mutation_rate:
                offspring = self.mutation_func(offspring)

    def replace(self, replace_func) -> None:
        self.population = replace_func(self.population, self.offsprings)

    def get_best(self) -> tuple:
        return self.population[0], self.fitness_scores[0]

    def average_fitness(self) -> float:
        return sum([i.fitness for i in self.population]) / len(self.population)

    def biodiversity(self) -> float:
        return len(set(self.population)) / len(self.population) * 100.0

    def run(self, generations):
        self.population = self.generation_func()
        self.evaluation(self.population)
