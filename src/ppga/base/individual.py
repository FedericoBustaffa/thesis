import sys


class Individual:
    def __init__(self, chromosome) -> None:
        self.chromosome = chromosome
        self.values = ()
        self.fitness = 0.0

    def __repr__(self) -> str:
        return f"{self.chromosome}: {self.fitness}"

    def __eq__(self, other) -> bool:
        assert isinstance(other, Individual)
        return self.chromosome == other.chromosome

    def __lt__(self, other) -> bool:
        assert isinstance(other, Individual)
        return self.fitness < other.fitness

    def __gt__(self, other) -> bool:
        assert isinstance(other, Individual)
        return self.fitness > other.fitness

    def __sizeof__(self) -> int:
        return sys.getsizeof(self.chromosome) + sys.getsizeof(self.fitness)
