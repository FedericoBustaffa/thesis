import random

from loguru import logger


class Replacer:
    def __init__(self, replace_func):
        self.replace_func = replace_func

    def perform(self, population, scores, offsprings, offsprings_scores):
        return self.replace_func(population, scores, offsprings, offsprings_scores)


if __name__ == "__main__":
    import sys

    import numpy as np
    from crossoverator import Crossoverator
    from evaluator import Evaluator
    from generator import Generator
    from mater import Mater
    from mutator import Mutator
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

    offsprings_scores = evaluator.perform(offsprings)

    def merge(population, scores1, offsprings, scores2) -> tuple:

        population = np.array(population)
        scores1 = np.array(scores1)
        offsprings = np.array(offsprings)
        scores2 = np.array(scores2)

        sort_indices = np.flip(np.argsort(scores1))
        population = np.array([population[i] for i in sort_indices])
        scores1 = scores1[sort_indices]

        sort_indices = np.flip(np.argsort(scores2))
        offsprings = np.array([offsprings[i] for i in sort_indices])
        scores2 = scores2[sort_indices]

        next_generation = np.zeros(population.shape, dtype=np.int64)
        next_gen_scores = np.zeros(scores1.shape, dtype=np.float64)
        index = 0
        index1 = 0
        index2 = 0

        while (
            index < len(population)
            and index1 < len(population)
            and index2 < len(offsprings)
        ):
            if scores1[index1] > scores2[index2]:
                next_generation[index] = population[index1]
                next_gen_scores[index] = scores1[index1]
                index1 += 1
            else:
                next_generation[index] = offsprings[index2]
                next_gen_scores[index] = scores2[index2]
                index2 += 1

            index += 1

        if index1 >= len(population):
            return next_generation, next_gen_scores
        elif index2 >= len(offsprings):
            next_generation[index:] = population[index1 : len(population) - index2]
            next_gen_scores[index:] = scores1[index1 : len(scores1) - index2]

        return next_generation, next_gen_scores

    replacer = Replacer(merge)
    population, scores = replacer.perform(
        population, scores, offsprings, offsprings_scores
    )

    logger.debug(f"population size {len(population)}")
