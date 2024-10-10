import sys


class Fitness:
    def __init__(self, weights: tuple) -> None:
        self.weights = weights
        self.values = None

    @property
    def fitness(self) -> float:
        if self.values is None:
            return 0.0
        return sum([v * w for v, w in zip(self.values, self.weights)])

    def __eq__(self, other) -> bool:
        assert isinstance(other, Fitness)
        return self.fitness == other.fitness

    def __lt__(self, other) -> bool:
        assert isinstance(other, Fitness)
        return self.fitness < other.fitness

    def __gt__(self, other) -> bool:
        assert isinstance(other, Fitness)
        return self.fitness > other.fitness

    def __sizeof__(self) -> int:
        return sys.getsizeof(self.weights) + sys.getsizeof(self.values)

    def __str__(self) -> str:
        return str(self.fitness)

    def __repr__(self) -> str:
        return str(self.fitness)


class Individual:
    def __init__(self, chromosome, fitness: Fitness) -> None:
        self.chromosome = chromosome
        self.fitness = fitness

    def __repr__(self) -> str:
        return f"{self.chromosome}: {self.fitness.fitness}"

    def __eq__(self, other) -> bool:
        assert isinstance(other, Individual)
        return self.chromosome == other.chromosome

    def __lt__(self, other) -> bool:
        assert isinstance(other, Individual)
        return self.fitness < other.fitness

    def __gt__(self, other) -> bool:
        assert isinstance(other, Individual)
        return self.fitness > other.fitness
