import multiprocessing as mp
import random
import time


class Algorithm:
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
        workers_num: int,
    ) -> None:

        self.population_size = population_size
        self.gen_func = gen_func
        self.fitness_func = fitness_func
        self.selection_func = selection_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.mutation_rate = mutation_rate
        self.replace_func = replace_func
        self.workers_num = workers_num

        self.population = []
        self.scores = []
        self.offsprings = []
        self.offsprings_scores = []
        self.couples = []

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

    def selection(self):
        start = time.perf_counter()
        self.selected = self.selection_func(self.scores)
        self.timings["selection"] += time.perf_counter() - start

    def mating(self):
        self.couples.clear()
        for _ in range(len(self.selected) // 2):
            father, mother = random.sample(self.selected, k=2)
            self.selected.remove(father)
            self.selected.remove(mother)

            self.couples.append([father, mother])

    def work(self, couples):
        offsprings = []
        scores = []

        for father, mother in couples:
            offspring1, offspring2 = self.crossover_func(
                self.population[father], self.population[mother]
            )

            if random.random() < self.mutation_rate:
                offspring1 = self.mutation_func(offspring1)

            if random.random() < self.mutation_rate:
                offspring2 = self.mutation_func(offspring2)

            score1 = self.fitness_func(offspring1)
            score2 = self.fitness_func(offspring2)

            offsprings.extend([offspring1, offspring2])
            scores.extend([score1, score2])

        return offsprings, scores

    def replace(self):
        start = time.perf_counter()
        self.population, self.scores = self.replace_func(
            self.population, self.scores, self.offsprings, self.offsprings_scores
        )
        self.timings["replacement"] += time.perf_counter() - start

    def run(self, max_generations: int) -> None:
        # initial population gen
        self.generate()

        self.best = self.population[0]
        self.best_score = self.scores[0]

        print(f"first best: {self.best_score}")

        with mp.Pool(self.workers_num) as executor:
            for g in range(max_generations):
                print(f"generation: {g+1}")
                print(executor)

                # selection
                self.selection()

                # mating
                self.mating()

                self.offsprings.clear()
                self.offsprings_scores.clear()

                # start paralle work
                chunk_size = len(self.couples) // mp.cpu_count()
                chunks = [
                    self.couples[i * chunk_size : i * chunk_size + chunk_size]
                    for i in range(self.workers_num)
                ]

                results = executor.map(self.work, chunks)

                for offsprings, scores in results:
                    self.offsprings.extend(offsprings)
                    self.offsprings_scores.extend(scores)

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
