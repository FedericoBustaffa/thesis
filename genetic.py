import random


class Genome:
    def __init__(self, chromosome: list, fitness: float = 0):
        self.chromosome = chromosome
        self.fitness = fitness

    def __repr__(self) -> str:
        return f"{self.chromosome}: {self.fitness}"

    def __lq__(self, other) -> bool:
        return self.fitness < other.fitness

    def __eq__(self, other) -> bool:
        return self.chromosome == other.chromosome and self.fitness == other.fitness

    def __hash__(self) -> int:
        return hash((tuple(self.chromosome), self.fitness))


def generate(size: int, gen_func, *args) -> list[Genome]:
    chromosomes = []
    for _ in range(size):
        c = gen_func(*args)
        while c in chromosomes:
            c = gen_func(*args)
        chromosomes.append(c)

    return [Genome(c) for c in chromosomes]


def evaluation(individuals: list[Genome], fitness, *args) -> list[Genome]:
    for offspring in individuals:
        offspring.fitness = fitness(offspring.chromosome, *args)

    return sorted(individuals, key=lambda x: x.fitness, reverse=True)


# tournament selection
def selection(population: list[Genome], selection_func, *args) -> list[Genome]:
    return selection_func(population)


# one point crossover without repetitions
def crossover(selected: list[Genome], crossover_func) -> list[Genome]:
    offsprings = []
    while len(selected) > 0:
        father, mother = random.choices(selected, k=2)
        # while father.chromosome == mother.chromosome:
        #     print("crossover conflict")
        #     mother = random.choice(selected)
        offspring1, offspring2 = crossover_func(father, mother)

        offsprings.extend([offspring1, offspring2])
        selected.remove(father)
        try:
            selected.remove(mother)
        except ValueError:
            pass

    return [Genome(child) for child in offsprings]


# rotation mutation
def mutation(
    offsprings: list[Genome], mutation_func, mutation_rate: float
) -> list[Genome]:
    indices = [i for i in range(len(offsprings[0].chromosome))]
    for offspring in offsprings:
        if random.random() < mutation_rate:
            offspring = mutation_func(offspring)

    return offsprings


def replace(
    population: list[Genome], offsprings: list[Genome], replace_func
) -> list[Genome]:
    return replace_func(population, offsprings)


def biodiversity(population: list[Genome]) -> float:
    return len(set(population)) / len(population) * 100.0
