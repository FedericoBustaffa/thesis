class Chromosome:
    def __init__(
        self, chromosome, fitness_values: tuple = (float,), weights: tuple = (1.0,)
    ) -> None:
        self._chromosome = chromosome
        self._fitness_values = fitness_values
        self._weights = weights

    def __repr__(self) -> str:
        return f"{self._chromosome}: {self._fitness_values}"

    def __eq__(self, other) -> bool:
        return self._chromosome == other.chromosome

    # TODO implementare una logica per ordinare chromosomi
    # con più valori di fitness anche
