from ppga.base.individual import Individual


class Statistics:
    def __init__(self):
        self.best = []
        self.mean = []
        self.worst = []

    def update_fitness(self, population: list[Individual]) -> None:
        valid_fitness = [i.fitness for i in population if not i.invalid()]
        self.best.append(max(valid_fitness))
        self.mean.append(sum(valid_fitness) / len(valid_fitness))
        self.worst.append(min(valid_fitness))
