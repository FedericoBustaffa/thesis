import random

from loguru import logger


class Crossoverator:
    def __init__(self, crossover_func, crossover_rate: float = 0.8):
        self.crossover_func = crossover_func
        self.crossover_rate = crossover_rate

    def perform(self, couples):
        offsprings = []
        for c in couples:
            if random.random() < self.crossover_rate:
                offsprings.extend(self.crossover_func(c[0], c[1]))

        return offsprings


if __name__ == "__main__":
    import sys

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

    crossoverator = Crossoverator(one_point)
    offsprings = crossoverator.perform(couples)

    logger.success(f"offsprings generated: {len(offsprings)}")

    # for o in offsprings:
    #     logger.info(o)
