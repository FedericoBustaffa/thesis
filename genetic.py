import random


class Genome:
    def __init__(self, chromosome: list, fitness: float = 0):
        self.chromosome = chromosome
        self.fitness = fitness

    def __repr__(self) -> str:
        return f"{self.chromosome}: {self.fitness}"

    def __lq__(self, other) -> bool:
        return self.fitness < other.fitness

    def __eq__(self, other) -> bool:
        return self.chromosome == other.chromosome and self.fitness == other.fitness

    def __hash__(self) -> int:
        return hash((tuple(self.chromosome), self.fitness))


class GeneticAlgorithm:
    def __init__(
        self,
        generation,
        generation_args,
        fitness_func,
        fitness_func_args,
        selection,
        crossover,
        mutation,
    ):
        self.generation = generation
        self.generation_args = generation_args

        self.fitness_func = fitness_func
        self.fitness_func_args = fitness_func_args

        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

    def run(self):
        pass

    def get(self):
        pass
