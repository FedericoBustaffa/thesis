class Chromosome:
    def __init__(self, values, fitness=0) -> None:
        self.values = values
        self.fitness = fitness

    def __repr__(self) -> str:
        return f"{self.values}: {self.fitness}"

    def __eq__(self, other) -> bool:
        return self.values == other.values

    def __hash__(self) -> int:
        return hash((tuple(self.values), self.fitness))
