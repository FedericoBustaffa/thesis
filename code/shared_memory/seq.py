import random
import time


class GeneticAlgorithm:
    def __init__(
        self,
        population_size,
        gen_func,
        fitness_func,
        selection_func,
        crossover_func,
        mutation_func,
        mutation_rate,
        replace_func,
    ) -> None:

        self.population_size = population_size
        self.gen_func = gen_func
        self.fitness_func = fitness_func
        self.selection_func = selection_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.mutation_rate = mutation_rate
        self.replace_func = replace_func

        # statistics
        self.average_fitness = []
        self.best_fitness = []
        self.biodiversity = []
        self.timings = {
            "generation": 0.0,
            "evaluation": 0.0,
            "selection": 0.0,
            "crossover": 0.0,
            "mutation": 0.0,
            "replacement": 0.0,
        }

    def generate(self):
        self.population = []
        self.scores = []

        start = time.perf_counter()
        for _ in range(self.population_size):
            chromosome = self.gen_func()
            while chromosome in self.population:
                chromosome = self.gen_func()

            self.population.append(chromosome)
            self.scores.append(self.fitness_func(chromosome))

        self.timings["generation"] += time.perf_counter() - start

        self.population, self.scores = (
            list(t)
            for t in zip(
                *sorted(
                    zip(self.population, self.scores), key=lambda x: x[1], reverse=True
                )
            )
        )

        # init for faster crossover
        chromosome_length = len(self.population[0])
        self.offsprings = [
            [0 for _ in range(chromosome_length)]
            for _ in range(self.population_size // 2)
        ]
        self.offsprings_scores = [0.0 for _ in range(len(self.offsprings))]

    def selection(self):
        start = time.perf_counter()
        self.selected = self.selection_func(self.scores)
        self.timings["selection"] += time.perf_counter() - start

    def crossover(self) -> None:
        start = time.perf_counter()

        for i in range(0, len(self.selected), 2):
            father_idx, mother_idx = random.choices(self.selected, k=2)

            father = self.population[father_idx]
            mother = self.population[mother_idx]

            offspring1, offspring2 = self.crossover_func(father, mother)
            self.offsprings[i] = offspring1
            self.offsprings[i + 1] = offspring2

            self.selected.remove(father_idx)
            try:
                self.selected.remove(mother_idx)
            except ValueError:
                pass

        end = time.perf_counter()
        self.timings["crossover"] += end - start

    def mutation(self):
        start = time.perf_counter()

        for offspring in self.offsprings:
            if random.random() < self.mutation_rate:
                offspring = self.mutation_func(offspring)

        end = time.perf_counter()
        self.timings["mutation"] += end - start

    def evaluation(self):
        start = time.perf_counter()

        for i in range(len(self.offsprings)):
            self.offsprings_scores[i] = self.fitness_func(self.offsprings[i])

        end = time.perf_counter()
        self.timings["evaluation"] += end - start

    def replace(self):
        start = time.perf_counter()

        self.population, self.scores = self.replace_func(
            self.population, self.scores, self.offsprings, self.offsprings_scores
        )

        self.timings["replacement"] += time.perf_counter() - start

    def run(self, max_generations: int) -> None:

        self.generate()

        self.best = self.population[0]
        self.best_score = self.scores[0]

        for g in range(max_generations):
            print(f"generation: {g+1}")

            # --- selection ---
            self.selection()
            self.crossover()
            self.mutation()
            self.evaluation()
            self.replace()

            if self.best_score < self.scores[0]:
                self.best = self.population[0]
                self.best_score = self.scores[0]

            self.average_fitness.append(sum(self.scores) / len(self.scores))
            self.best_fitness.append(self.best_score)

            self.biodiversity.append(
                len(set(tuple(tuple(i) for i in self.population)))
                / len(self.population)
                * 100.0
            )

            # convergence check
            if self.best_score <= self.average_fitness[-1]:
                print(f"stop at generation {g+1}")
                print(f"best score: {self.best_score}")
                print(f"average fitness: {self.average_fitness[-1]}")
                break
