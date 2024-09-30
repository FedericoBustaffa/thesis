from loguru import logger


class Selector:
    def __init__(self, selection_func):
        self.selection_func = selection_func

    def perform(self, population, scores) -> list[int]:
        return self.selection_func(population, scores)


if __name__ == "__main__":
    import random
    import sys

    from evaluator import Evaluator
    from generator import Generator

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

    for i in range(len(chosen)):
        print(f"{chosen[i]}: {scores[i]}")
