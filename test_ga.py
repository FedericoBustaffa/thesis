import math
import random
import sys

import pandas as pd

from genetic import GeneticAlgorithm


class Town:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def generate(length):
    chromosome = [i for i in range(length)]
    random.shuffle(chromosome)

    return chromosome


def distance(t1: Town, t2: Town) -> float:
    return math.sqrt(math.pow(t1.x - t2.x, 2) + math.pow(t1.y - t2.y, 2))


def fitness(chromosome: list[int], towns: list[Town]) -> float:
    total_distance = 0
    for i in range(len(towns) - 1):
        total_distance += distance(towns[chromosome[i]], towns[chromosome[i + 1]])

    return 1 / total_distance


# tournament selection
def tournament(population):
    selected = []
    indices = [i for i in range(len(population))]

    for _ in range(len(population) // 2):
        first, second = random.choices(indices, k=2)
        # while first == second:
        #     print("selection conflict")
        #     second = random.choice(indices)

        if population[first].fitness > population[second].fitness:
            selected.append(population[first])
        else:
            selected.append(population[second])

        indices.remove(first)
        try:
            indices.remove(second)
        except ValueError:
            pass

    return selected


# one point crossover without repetitions
def one_point(selected: list[Genome]) -> list[Genome]:
    offsprings = []
    while len(selected) > 0:
        father, mother = random.choices(selected, k=2)
        # while father.chromosome == mother.chromosome:
        #     print("crossover conflict")
        #     mother = random.choice(selected)

        crossover_point = random.randint(1, len(father.chromosome) - 2)
        offspring1 = father.chromosome[:crossover_point]
        offspring2 = father.chromosome[crossover_point:]

        for gene in mother.chromosome:
            if gene not in offspring1:
                offspring1.append(gene)
            else:
                offspring2.append(gene)

        offsprings.extend([offspring1, offspring2])
        selected.remove(father)
        try:
            selected.remove(mother)
        except ValueError:
            pass

    return [Genome(child) for child in offsprings]


# rotation mutation
def rotation(offsprings: list[Genome], mutation_rate: float) -> list[Genome]:
    indices = [i for i in range(len(offsprings[0].chromosome))]
    for child in offsprings:
        if random.random() < mutation_rate:
            a, b = random.choices(indices, k=2)
            # while a == b:
            #     print("mutation conflict")
            #     b = random.choice(indices)
            first = a if a < b else b
            second = a if a > b else b

            head = child.chromosome[:first]
            middle = reversed(child.chromosome[first:second])
            tail = child.chromosome[second:]
            head.extend(middle)
            head.extend(tail)
            child.chromosome = head

    return offsprings


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"USAGE: py {sys.argv[0]} <T> <N> <G> <M>")
        exit(1)

    data = pd.read_csv(sys.argv[1])
    x = data["x"]
    y = data["y"]
    towns = [Town(x.iloc[i], y.iloc[i]) for i in range(len(data))]
    distances = [[distance(t1, t2) for t2 in towns] for t1 in towns]

    # Initial population size
    N = int(sys.argv[2])

    # Max generations
    G = int(sys.argv[3])

    # Mutation rate
    M = float(sys.argv[4])

    # useful data
    average_fitness = []
    best_fitness = []
    biodiversities = []
    timings = {
        "generation": 0.0,
        "evaluation": 0.0,
        "selection": 0.0,
        "crossover": 0.0,
        "mutation": 0.0,
        "replacement": 0.0,
    }

    ga = GeneticAlgorithm(
        generation=generate,
        generation_args=[len(towns)],
        fitness_func=fitness,
        fitness_func_args=[towns],
        selection=tournament,
        crossover=one_point,
        mutation=rotation,
    )

    ga.run()
    best = ga.get()
