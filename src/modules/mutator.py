import random

from loguru import logger


class Mutator:
    def __init__(self, mutation_func, mutation_rate: float = 0.2):
        self.mutation_func = mutation_func
        self.mutation_rate = mutation_rate

    def perform(self, population):
        for i in range(len(population)):
            if random.random() < self.mutation_rate:
                population[i] = self.mutation_func(population[i])

        return population


if __name__ == "__main__":
    import sys

    from crossoverator import Crossoverator
    from evaluator import Evaluator
    from generator import Generator
    from mater import Mater
    from selector import Selector

    if len(sys.argv) != 2:
        logger.error(f"USAGE: py {sys.argv[0]} <size>")
        exit(1)

    size = int(sys.argv[1])
    generator = Generator(size, lambda: [random.randint(0, 1) for _ in range(size)])
    population = generator.perform()

    evaluator = Evaluator(lambda x: sum(x))
    scores = evaluator.perform(population)

    def tournament(population, scores):
        selected = []
        indices = [i for i in range(len(scores))]

        for _ in range(len(scores) // 2):
            first, second = random.choices(indices, k=2)
            while first == second:
                first, second = random.choices(indices, k=2)

            if scores[first] > scores[second]:
                selected.append(population[first])
                indices.remove(first)
            else:
                selected.append(population[second])
                indices.remove(second)

        return selected

    selector = Selector(tournament)
    chosen = selector.perform(population, scores)

    def mating_func(chosen):
        indices = [i for i in range(len(chosen))]
        couples = []
        for _ in range(len(chosen) // 2):
            father, mother = random.sample(indices, k=2)
            couples.append([chosen[father], chosen[mother]])
            indices.remove(father)
            indices.remove(mother)

        return couples

    mater = Mater(mating_func)
    couples = mater.perform(chosen)

    def one_point(father, mother):
        point = random.randint(1, len(father) - 2)

        offspring1 = father[:point] + mother[point:]
        offspring2 = father[point:] + mother[:point]

        return [offspring1, offspring2]

    crossoverator = Crossoverator(one_point, 1.0)
    offsprings = crossoverator.perform(couples)

    def swap(chromosome):
        i, j = random.sample([i for i in range(len(chromosome))], k=2)
        temp = chromosome[i]
        chromosome[i] = chromosome[j]
        chromosome[j] = temp

        return chromosome

    mutator = Mutator(swap, 0.2)
    offsprings = mutator.perform(offsprings)
