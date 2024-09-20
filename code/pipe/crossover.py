import random
import time


def mating(self):
    start = time.perf_counter()
    couples = []
    for _ in range(0, len(self.selected), 2):
        try:
            father, mother = random.sample(self.selected, k=2)
            couples.append(
                (self.population[father].values, self.population[mother].values)
            )

            self.selected.remove(father)
            self.selected.remove(mother)
        except:
            pass
    self.timings["crossover"] += time.perf_counter() - start

    return couples
