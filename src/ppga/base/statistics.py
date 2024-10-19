import time


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

    def __setitem__(self, key: str, value: float):
        self.timings[key] = value

    def __getitem__(self, key: str) -> float:
        return self.timings[key]

    def add_time(self, field: str, start: float) -> None:
        self.timings[field] += time.perf_counter() - start

    def cme(self) -> float:
        return sum(
            [
                self.timings["crossover"],
                self.timings["mutation"],
                self.timings["evaluation"],
            ]
        )

    def reset(self) -> None:
        for k in self.timings.keys():
            self.timings[k] = 0.0

    def push_best(self, fitness: float) -> None:
        self.best.append(fitness)

    def push_worst(self, fitness: float) -> None:
        self.worst.append(fitness)
