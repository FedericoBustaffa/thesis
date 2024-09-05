import random


class Genome:
    def __init__(self, chromosome: list, fitness: float = 0):
        self.chromosome = chromosome
        self.fitness = fitness

    def __repr__(self) -> str:
        return f"{self.chromosome}: {self.fitness}"

    def __lq__(self, other) -> bool:
        return self.fitness < other.fitness


def generate(population_size: int, gen_func, *args):
    return [Genome(gen_func(args)) for _ in range(population_size)]
