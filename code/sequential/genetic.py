import random
import time


class Chromosome:

    def __init__(self, values: list, fitness: float = 0.0):
        self.values = values
        self.fitness = fitness

    def __eq__(self, other) -> bool:
        return self.values == other.values

    def __hash__(self) -> int:
        return hash((tuple(self.values), self.fitness))

    def __repr__(self) -> str:
        return f"{self.values}: {self.fitness:.3f}"


class GeneticAlgorithm:

    def __init__(
        self,
        population_size: int,
        generation_func,
        fitness_func,
        selection_func,
        crossover_func,
        mutation_func,
        mutation_rate,
        replace_func,
    ):
        self.population_size = population_size
        self.generation_func = generation_func
        self.fitness_func = fitness_func
        self.selection_func = selection_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.mutation_rate = mutation_rate
        self.replace_func = replace_func

        # statistics
        self.average_fitness = []
        self.best_fitness = []
        self.biodiversities = []
        self.timings = {
            "generation": 0.0,
            "evaluation": 0.0,
            "selection": 0.0,
            "crossover": 0.0,
            "crossover_operator": 0.0,
            "mutation": 0.0,
            "replacement": 0.0,
            "stuff": 0.0,
        }

    def generation(self) -> None:
        start = time.perf_counter()

        chromosomes = []
        for _ in range(self.population_size):
            c = self.generation_func()
            while c in chromosomes:
                c = self.generation_func()

            chromosomes.append(c)

        end = time.perf_counter()
        self.timings["generation"] += end - start

        start = time.perf_counter()
        scores = list(map(self.fitness_func, chromosomes))

        self.population = sorted(
            [
                Chromosome(chromosome, score)
                for chromosome, score in zip(chromosomes, scores)
            ],
            key=lambda x: x.fitness,
            reverse=True,
        )
        end = time.perf_counter()
        self.timings["evaluation"] += end - start

        # init for faster crossover
        chromosome_length = len(chromosomes[0])
        self.offsprings = [
            Chromosome([0 for _ in range(chromosome_length)])
            for _ in range(self.population_size // 2)
        ]

    def selection(self) -> None:
        start = time.perf_counter()
        self.selected = self.selection_func(self.population)
        end = time.perf_counter()
        self.timings["selection"] += end - start

    def crossover(self) -> None:
        start = time.perf_counter()
        for i in range(0, len(self.selected), 2):
            father_idx, mother_idx = random.choices(self.selected, k=2)

            father = self.population[father_idx].values
            mother = self.population[mother_idx].values

            nano_start = time.perf_counter_ns()
            offspring1, offspring2 = self.crossover_func(father, mother)
            self.offsprings[i] = Chromosome(offspring1)
            self.offsprings[i + 1] = Chromosome(offspring2)
            nano_end = time.perf_counter_ns()
            if self.timings["crossover_operator"] < (nano_end - nano_start):
                self.timings["crossover_operator"] = nano_end - nano_start

            self.selected.remove(father_idx)
            try:
                self.selected.remove(mother_idx)
            except ValueError:
                pass

        end = time.perf_counter()
        self.timings["crossover"] += end - start

    def mutation(self) -> None:
        start = time.perf_counter()
        for offspring in self.offsprings:
            if random.random() < self.mutation_rate:
                offspring.values = self.mutation_func(offspring.values)
        end = time.perf_counter()
        self.timings["mutation"] += end - start

    def evaluation(self) -> None:
        start = time.perf_counter()
        for i in self.offsprings:
            i.fitness = self.fitness_func(i.values)

        self.offsprings = sorted(self.offsprings, key=lambda x: x.fitness, reverse=True)
        end = time.perf_counter()
        self.timings["evaluation"] += end - start

    def replace(self) -> None:
        start = time.perf_counter()
        self.population = self.replace_func(self.population, self.offsprings)
        end = time.perf_counter()
        self.timings["evaluation"] += end - start

    def get_best(self) -> Chromosome:
        return self.best

    def get_average_fitness(self) -> list[float]:
        return self.average_fitness

    def get_best_fitness(self) -> list[float]:
        return self.best_fitness

    def get_biodiversity(self) -> list[float]:
        return self.biodiversities

    def get_timings(self) -> dict[str, float]:
        return self.timings

    def run(self, generations):

        self.generation()

        self.best = self.population[0]
        print(f"first best score: {self.best.fitness}")

        for g in range(generations):
            print(f"generation: {g+1}")

            self.selection()
            self.crossover()
            self.mutation()
            self.evaluation()
            self.replace()

            start = time.perf_counter()
            if self.best.fitness < self.population[0].fitness:
                self.best = self.population[0]

            self.average_fitness.append(
                sum([i.fitness for i in self.population]) / len(self.population)
            )

            self.biodiversities.append(
                len(list(set(self.population))) / len(self.population) * 100.0
            )

            self.best_fitness.append(self.best.fitness)
            self.timings["stuff"] += time.perf_counter() - start

            # convergence check
            # if self.best.fitness <= self.average_fitness[-1]:
            #     print(f"stop at generation {g+1}")
            #     break
