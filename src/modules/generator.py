import random
import sys

from loguru import logger


class Generator:
    def __init__(self, population_size: int, generation_func) -> None:
        self.population_size = population_size
        self.generation_func = generation_func

    def perform(self) -> list:
        population = []
        for _ in range(self.population_size):
            chromosome = self.generation_func()
            # while chromosome in population:
            #     chromosome = self.generation_func()
            population.append(chromosome)

        return population


if __name__ == "__main__":
    # simple test main

    if len(sys.argv) < 2:
        logger.error(f"USAGE: py {sys.argv[0]} <population_size>")
        logger.error(f"exit with error {1}")
        exit(1)

    N = int(sys.argv[1])

    generator = Generator(N, lambda: [random.randint(0, 1) for _ in range(5)])

    # allows repetitions
    print("------- 1st population -------")
    population = generator.perform()
    for i in population:
        print(i)

    # permutation
    def permutation():
        chromosome = [i for i in range(5)]
        random.shuffle(chromosome)

        return chromosome

    print("------- 2nd population -------")
    generator.generation_func = permutation
    population = generator.perform()
    for i in population:
        print(i)
