from loguru import logger


class Evaluator:
    def __init__(self, fitness_func) -> None:
        self.fitness_func = fitness_func

    def perform(self, population):
        scores = [0.0 for _ in range(len(population))]
        for i in range(len(population)):
            scores[i] = self.fitness_func(population[i])

        return scores


if __name__ == "__main__":

    def fitness(chromosome):
        return sum(chromosome)
