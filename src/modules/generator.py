import random
import sys
import time

from loguru import logger


class Generator:
    def __init__(self, population_size: int, generation_func) -> None:
        self.population_size = population_size
        self.generation_func = generation_func

    def perform(self) -> list:
        population = []
        start = time.perf_counter()
        for _ in range(self.population_size):
            chromosome = self.generation_func()
            # while chromosome in population:
            #     chromosome = self.generation_func()
            population.append(chromosome)

        end = time.perf_counter()
        logger.debug(
            f"{len(population)} chromosomes generated in {end - start:.6f} seconds"
        )

        return population


if __name__ == "__main__":
    # simple test main

    if len(sys.argv) < 3:
        logger.error(
            f"USAGE: py {sys.argv[0]} <population_size> <chromosome_length> <log_level=DEBUG>"
        )
        logger.error(f"exit with error {1}")
        exit(1)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | {file}:{line} | <level>{level} - {message}</level>",
        level=sys.argv[3].upper() if len(sys.argv) == 4 else "DEBUG",
    )

    N = int(sys.argv[1])
    L = int(sys.argv[2])

    generator = Generator(N, lambda: [random.randint(0, 1) for _ in range(L)])

    # allows repetitions
    logger.trace("------- 1st population -------")
    population = generator.perform()
    for i in population:
        logger.trace(i)

    # permutation
    def permutation():
        chromosome = [i for i in range(L)]
        random.shuffle(chromosome)

        return chromosome

    logger.trace("------- 2nd population -------")
    generator.generation_func = permutation
    population = generator.perform()
    for i in population:
        logger.trace(i)
