import time


def select(self):
    start = time.perf_counter()
    self.selected = self.selection_func(self.population)
    self.timings["selection"] += time.perf_counter() - start
