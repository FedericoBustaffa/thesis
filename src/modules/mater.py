from loguru import logger


class Mater:
    def __init__(self, mating_func):
        self.mating_func = mating_func

    def perform(self, chosen):
        return self.mating_func(chosen)


if __name__ == "__main__":
    import random
    import sys

    from evaluator import Evaluator
    from generator import Generator
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

    for c in couples:
        print(f"{c[0]} - {c[1]}")
