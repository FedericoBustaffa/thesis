import random

from modules import Fitness, Individual


class ToolBox:
    def __init__(self) -> None:
        pass

    def set_fitness_weights(self, weights: tuple) -> None:
        self.weights = weights

    def set_generation(self, func, *args, **kwargs) -> None:
        self.generation_func = func
        self.generation_args = args
        self.generation_kwargs = kwargs

    def generate(self, population_size) -> list[Individual]:
        population = []
        for _ in range(population_size):
            chromosome = self.generation_func(
                *self.generation_args, **self.generation_kwargs
            )
            while chromosome in population:
                chromosome = self.generation_func(
                    *self.generation_args, **self.generation_kwargs
                )

            population.append(
                self.generation_func(*self.generation_args, **self.generation_kwargs)
            )

        return [Individual(c, Fitness(self.weights)) for c in population]

    def set_selection(self, func, *args, **kwargs) -> None:
        self.selection_func = func
        self.selection_args = args
        self.selection_kwargs = kwargs

    def select(self, population: list[Individual]) -> list[Individual]:
        return self.selection_func(
            population, *self.selection_args, **self.selection_kwargs
        )

    def set_mating(self, func, *args, **kwargs) -> None:
        self.mating_func = func
        self.mating_args = args
        self.mating_kwargs = kwargs

    def mate(self, population: list[Individual]) -> list[tuple[Individual]]:
        return self.mating_func(population, *self.mating_args, **self.mating_kwargs)

    def set_crossover(self, func, rate: float = 0.8, *args, **kwargs) -> None:
        self.crossover_func = func
        self.crossover_rate = rate
        self.crossover_args = args
        self.crossover_kwargs = kwargs

    def crossover(self, couples: list[tuple[Individual]]) -> list[Individual]:
        offsprings = []
        for c in couples:
            if random.random() < self.crossover_rate:
                new_offsprings = list(
                    self.crossover_func(
                        c[0].chromosome,
                        c[1].chromosome,
                        *self.crossover_args,
                        **self.crossover_kwargs,
                    )
                )
                offsprings.extend(new_offsprings)

        return [Individual(o, Fitness(self.weights)) for o in offsprings]

    def set_mutation(self, func, rate: float = 0.2, *args, **kwargs):
        self.mutation_func = func
        self.mutation_rate = rate
        self.mutation_args = args
        self.mutation_kwargs = kwargs

    def mutate(self, population: list[Individual]) -> list[Individual]:
        for i in population:
            if random.random() < self.mutation_rate:
                i.chromosome = self.mutation_func(
                    i.chromosome, *self.mutation_args, **self.mutation_kwargs
                )

        return population

    def set_evaluation(self, func, *args, **kwargs) -> None:
        self.evaluation_func = func
        self.evaluation_args = args
        self.evaluation_kwargs = kwargs

    def evaluate(self, population: list[Individual]) -> None:
        for i in population:
            i.fitness.values = self.evaluation_func(
                i.chromosome, *self.evaluation_args, **self.evaluation_kwargs
            )

        return population

    def set_replacement(self, func, *args, **kwargs):
        self.replacement_func = func
        self.replacement_args = args
        self.replacement_kwargs = kwargs

    def replace(
        self, population: list[Individual], offsprings: list[Individual]
    ) -> list[Individual]:
        return self.replacement_func(population, offsprings)
        return self.replacement_func(population, offsprings)
        return self.replacement_func(population, offsprings)
        return self.replacement_func(population, offsprings)
