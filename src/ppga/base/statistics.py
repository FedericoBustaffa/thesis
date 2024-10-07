class Statistics:
    def __init__(self):
        self.timings = {
            "generation": 0.0,
            "evaluation": 0.0,
            "selection": 0.0,
            "mating": 0.0,
            "crossover": 0.0,
            "mutation": 0.0,
            "replacement": 0.0,
            "parallel": 0.0,
            "synchronization": 0.0,
        }

        self.best = []
        self.worst = []

    def add_time(self, field: str, time: float) -> None:
        self.timings[field] += time

    def push_best(self, fitness: float) -> None:
        self.best.append(fitness)

    def push_worst(self, fitness: float) -> None:
        self.worst.append(fitness)
