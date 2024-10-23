from ppga.base.individual import Individual


class HallOfFame:
    def __init__(self, size: int):
        self.size = size
        self.best = []

    def __getitem__(self, index: int) -> Individual:
        return self.best[index]

    def update(self, population: list[Individual]):
        self.best = sorted(list(set(self.best + population)), reverse=True)[: self.size]

    def clear(self):
        self.best.clear()
