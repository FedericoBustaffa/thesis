import random
import time
from functools import partial

from ppga.base.individual import Individual
from ppga.tools.coupling import couples_mating
from ppga.tools.replacement import total


class ToolBox:
    def __init__(self):
        self.mating_func = couples_mating
        self.mating_args = ()
        self.mating_kwargs = {}

        self.replacement_func = total
        self.replacement_args = ()
        self.replacement_kwargs = {}

    def set_weights(self, weights: tuple) -> None:
        self.weights = weights

    def set_attributes(self, func, *args, **kwargs) -> None:
        self.attrib_func = func
        self.attrib_args = args
        self.attrib_kwargs = kwargs

    def set_generation(self, func, *args, **kwargs) -> None:
        self.generation_func = func
        self.generation_args = args
        self.generation_kwargs = kwargs

    def generate(self, population_size) -> list[Individual]:
        attribute_generator = partial(
            self.attrib_func, *self.attrib_args, **self.attrib_kwargs
        )

        population = self.generation_func(
            attribute_generator,
            population_size,
            *self.generation_args,
            **self.generation_kwargs,
        )

        return [Individual(c) for c in population]

    def set_selection(self, func, *args, **kwargs) -> None:
        self.selection_func = func
        self.selection_args = args
        self.selection_kwargs = kwargs

    def select(
        self, population: list[Individual], population_size: int
    ) -> list[Individual]:
        return self.selection_func(
            population, population_size, *self.selection_args, **self.selection_kwargs
        )

    def set_mating(self, func, *args, **kwargs) -> None:
        self.mating_func = func
        self.mating_args = args
        self.mating_kwargs = kwargs

    def mate(self, population: list[Individual]) -> list[tuple]:
        return self.mating_func(population, *self.mating_args, **self.mating_kwargs)

    def set_crossover(self, func, cxpb: float = 0.8, *args, **kwargs) -> None:
        self.crossover_func = func
        self.crossover_pb = cxpb
        self.crossover_args = args
        self.crossover_kwargs = kwargs

    def crossover(self, couples: list[tuple]) -> list[Individual]:
        offsprings = []
        for c in couples:
            if random.random() < self.crossover_pb:
                new_offsprings = list(
                    self.crossover_func(
                        c[0].chromosome,
                        c[1].chromosome,
                        *self.crossover_args,
                        **self.crossover_kwargs,
                    )
                )
                offsprings.extend(new_offsprings)

        return [Individual(o) for o in offsprings]

    def set_mutation(self, func, mutpb: float = 0.2, *args, **kwargs):
        self.mutation_func = func
        self.mutation_pb = mutpb
        self.mutation_args = args
        self.mutation_kwargs = kwargs

    def mutate(self, population: list[Individual]) -> list[Individual]:
        for i in population:
            if random.random() < self.mutation_pb:
                i.chromosome = self.mutation_func(
                    i.chromosome, *self.mutation_args, **self.mutation_kwargs
                )

        return population

    def set_evaluation(self, func, *args, **kwargs) -> None:
        self.evaluation_func = func
        self.evaluation_args = args
        self.evaluation_kwargs = kwargs

    def evaluate(self, population: list[Individual]) -> tuple[list[Individual], float]:
        times = []
        for i in population:
            start = time.perf_counter()
            i.values = self.evaluation_func(
                i.chromosome, *self.evaluation_args, **self.evaluation_kwargs
            )
            i.fitness = sum([v * w for v, w in zip(i.values, self.weights)])
            times.append(time.perf_counter() - start)

        try:
            mean = sum(times) / len(times)
            return population, mean
        except ZeroDivisionError:
            return population, 0.0

    def set_replacement(self, func, *args, **kwargs):
        self.replacement_func = func
        self.replacement_args = args
        self.replacement_kwargs = kwargs

    def replace(
        self, population: list[Individual], offsprings: list[Individual]
    ) -> list[Individual]:
        return self.replacement_func(population, offsprings)
