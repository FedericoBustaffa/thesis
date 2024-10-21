from ppga.base.individual import Individual


class HallOfFame:
    def __init__(self, size: int):
        self.size = size
        self.hof = []

    def __getitem__(self, index: int) -> Individual:
        return self.hof[index]

    def update(self, population: list[Individual]):
        if len(self.hof) == 0:
            self.hof = population[: self.size]
        else:
            j = 0
            for i in range(self.size):
                if population[i] > self.hof[j]:
                    self.hof[j] = population[i]
                    j += 1
