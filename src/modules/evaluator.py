import time

from loguru import logger


class Evaluator:
    def __init__(self, fitness_func) -> None:
        self.fitness_func = fitness_func

    def perform(self, population):
        scores = [0.0 for _ in range(len(population))]

        start = time.perf_counter()
        for i in range(len(population)):
            scores[i] = self.fitness_func(population[i])
        end = time.perf_counter()
        logger.trace(
            f"evaluation of {len(population)} done in {end - start:.6f} seconds"
        )

        return scores


if __name__ == "__main__":
    import random
    import sys

    from generator import Generator

    if len(sys.argv) != 2:
        logger.error(f"USAGE: py {sys.argv[0]} <size>")
        exit(1)

    size = int(sys.argv[1])
    generator = Generator(size, lambda: [random.randint(0, 1) for _ in range(size)])
    population = generator.perform()

    evaluator = Evaluator(lambda x: sum(x))
    scores = evaluator.perform(population)

    for i in range(size):
        print(f"{population[i]}: {scores[i]}")
