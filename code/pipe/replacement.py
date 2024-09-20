import time


def replace(self):
    start = time.perf_counter()
    self.population = self.replace_func(self.population, self.offsprings)
    self.timings["replacement"] += time.perf_counter() - start
