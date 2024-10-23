import time


class Statistics:
    def __init__(self):
        self.timings = {
            "generation": [],
            "evaluation": [],
            "selection": [],
            "mating": [],
            "crossover": [],
            "mutation": [],
            "replacement": [],
            "parallel": [],
            "synchronization": [],
        }

        self.best = []
        self.worst = []

    def __setitem__(self, key: str, value: float):
        self.timings[key].append(value)

    def __getitem__(self, key: str) -> float:
        return sum(self.timings[key])

    def add_time(self, field: str, start: float) -> None:
        self.timings[field].append(time.perf_counter() - start)

    def cme(self) -> float:
        return sum(
            [
                sum(self.timings["crossover"]),
                sum(self.timings["mutation"]),
                sum(self.timings["evaluation"]),
            ]
        )

    def reset(self) -> None:
        for k in self.timings.keys():
            self.timings[k] = []

    def push_best(self, fitness: float) -> None:
        self.best.append(fitness)

    def push_worst(self, fitness: float) -> None:
        self.worst.append(fitness)
