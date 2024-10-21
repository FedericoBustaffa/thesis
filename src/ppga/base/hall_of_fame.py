from ppga.base.individual import Individual


class HallOfFame:
    def __init__(self, size: int):
        self.size = size
        self.best = []

    def __getitem__(self, index: int) -> Individual:
        return self.best[index]

    def update(self, population: list[Individual]):
        uniques = sorted(list(set(population)), reverse=True)[: self.size]
        self.best = sorted(list(set(self.best + uniques)), reverse=True)[: self.size]

    def clear(self):
        self.best.clear()
