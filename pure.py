import random


class Genome:
    def __init__(self, chromosome: list, fitness: float = 0):
        self.chromosome = chromosome
        self.fitness = fitness

    def __repr__(self) -> str:
        return f"{self.chromosome}: {self.fitness}"

    def __lq__(self, other) -> bool:
        return self.fitness < other.fitness


def chromosome_generation(T: int) -> list[int]:
    chromosome = [i for i in range(T)]
    random.shuffle(chromosome)

    return chromosome


def generate(population_size: int, chromosome_length: int) -> list[Genome]:
    chromosomes = []
    for _ in range(population_size):
        c = chromosome_generation(chromosome_length)
        while c in chromosomes:
            c = chromosome_generation(chromosome_length)
        chromosomes.append(c)

    return [Genome(c) for c in chromosomes]


# tournament selection
def selection(population: list[Genome]) -> list[Genome]:
    selected = []
    indices = [i for i in range(len(population))]

    for _ in range(len(population) // 2):
        first, second = random.choices(indices, k=2)
        while first == second:
            # print("selection conflict")
            second = random.choice(indices)

        if population[first].fitness > population[second].fitness:
            selected.append(population[first])
        else:
            selected.append(population[second])

        indices.remove(first)
        indices.remove(second)

    return selected


# one point crossover without repetitions
def crossover(selected: list[Genome]) -> list[Genome]:
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


def mutation(offsprings: list[Genome], mutation_rate: float) -> list[Genome]:
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


def evaluation(individuals: list[Genome], fitness, towns) -> list[Genome]:
    for offspring in individuals:
        offspring.fitness = fitness(offspring.chromosome, towns)

    return sorted(individuals, key=lambda x: x.fitness, reverse=True)


def replace(population: list[Genome], offsprings: list[Genome]) -> list[Genome]:
    next_generation = []
    index1 = 0
    index2 = 0

    while index1 < len(population) and index2 < len(offsprings):
        if population[index1].fitness > offsprings[index2].fitness:
            next_generation.append(population[index1])
            index1 += 1
        else:
            next_generation.append(offsprings[index2])
            index2 += 1

    if index1 >= len(population):
        next_generation.extend(offsprings[index2:])
    else:
        next_generation.extend(population[index1:])

    return next_generation[: len(population)]
