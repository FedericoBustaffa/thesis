import math
import multiprocessing as mp
import multiprocessing.connection as conn
import random
import time


class Chromosome:
    def __init__(self, values, fitness=0) -> None:
        self.values = values
        self.fitness = fitness

    def __repr__(self) -> str:
        return f"{self.values}: {self.fitness}"

    def __eq__(self, other) -> bool:
        return self.values == other.values

    def __hash__(self) -> int:
        return hash((tuple(self.values), self.fitness))


class SharedMemoryGeneticAlgorithm:

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
        num_of_workers: int = mp.cpu_count(),
    ) -> None:

        self.population_size = population_size
        self.gen_func = gen_func
        self.fitness_func = fitness_func
        self.selection_func = selection_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.mutation_rate = mutation_rate
        self.replace_func = replace_func

        self.population = []
        self.offsprings = [Chromosome([]) for _ in range(population_size // 2)]

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

        # processing
        self.pipes = [mp.Pipe() for _ in range(num_of_workers)]
        self.workers = [
            mp.Process(target=self.work, args=[self.pipes[i][1]])
            for i in range(num_of_workers)
        ]

        for w in self.workers:
            w.start()

    def generate(self):
        start = time.perf_counter()
        chromosomes = []
        for _ in range(self.population_size):
            values = self.gen_func()
            while values in chromosomes:
                values = self.gen_func()

            chromosomes.append(values)

        self.population = [
            Chromosome(values, self.fitness_func(values)) for values in chromosomes
        ]

        self.timings["generation"] += time.perf_counter() - start

    def selection(self):
        start = time.perf_counter()
        self.selected = self.selection_func(self.population)
        self.timings["selection"] += time.perf_counter() - start

    def make_couples(self):
        start = time.perf_counter()
        couples = []
        for _ in range(0, len(self.selected), 2):
            try:
                father, mother = random.sample(self.selected, k=2)
                couples.append(
                    (self.population[father].values, self.population[mother].values)
                )

                self.selected.remove(father)
                self.selected.remove(mother)
            except:
                pass
        self.timings["crossover"] += time.perf_counter() - start

        return couples

    def work(self, pipe: conn.Connection):
        timings = {"crossover": 0.0, "mutation": 0.0, "evaluation": 0.0}
        couples = pipe.recv()
        while couples != None:
            for i in range(len(couples)):
                father, mother = couples[i]
                start = time.perf_counter()
                o1, o2 = self.crossover_func(father, mother)
                timings["crossover"] += time.perf_counter() - start
                # print(f"{mp.current_process().name} offsprings: {o1}, {o2}")

                start = time.perf_counter()
                o1 = self.mutation_func(o1)
                o2 = self.mutation_func(o2)
                timings["mutation"] += time.perf_counter() - start
                # print(f"{mp.current_process().name} mutated offsprings: {o1}, {o2}")

                start = time.perf_counter()
                self.offsprings[i * 2] = Chromosome(o1, self.fitness_func(o1))
                self.offsprings[i * 2 + 1] = Chromosome(o2, self.fitness_func(o2))
                timings["evaluation"] += time.perf_counter() - start

            pipe.send(self.offsprings)
            couples = pipe.recv()

        pipe.send(timings)
        pipe.close()

    def replace(self):
        start = time.perf_counter()
        self.population = self.replace_func(self.population, self.offsprings)
        self.timings["replacement"] += time.perf_counter() - start

    def run(self, max_generations: int):

        # generate random population
        self.generate()
        # print("generated")
        # for i in self.population:
        #     print(i)

        self.best = self.population[0]
        print(f"first best: {self.best.fitness}")

        for g in range(max_generations):
            # select individuals for crossover
            self.selection()
            # print("selected")
            # for i in self.selected:
            #     print(f"{i}. {self.population[i]}")

            # creating couples for crossover
            couples = self.make_couples()
            # print("couples")
            # for c in couples:
            #     print(c)

            # sending couples to processes
            portion = math.ceil(len(couples) / len(self.workers))
            for i in range(len(self.workers)):
                self.pipes[i][0].send(couples[i * portion : i * portion + portion])

            self.offsprings.clear()
            for pipe in self.pipes:
                partial_offsprings = pipe[0].recv()
                if partial_offsprings != None:
                    self.offsprings.extend(partial_offsprings)
                else:
                    continue

            # print("offsprings")
            # for i in self.offsprings:
            #     print(i)

            self.replace()

            if self.best.fitness < self.population[0].fitness:
                self.best = self.population[0]

            self.average_fitness.append(
                sum([i.fitness for i in self.population]) / len(self.population)
            )

            self.biodiversity.append(
                len(list(set(self.population))) / len(self.population) * 100.0
            )

            self.best_fitness.append(self.population[0].fitness)

            # convergence check
            if self.best.fitness <= self.average_fitness[-1]:
                print(f"stop at generation {g}")
                break

        for i in range(len(self.workers)):
            self.pipes[i][0].send(None)
            worker_timings = self.pipes[i][0].recv()
            self.timings["crossover"] += worker_timings["crossover"]
            self.timings["mutation"] += worker_timings["mutation"]
            self.timings["evaluation"] += worker_timings["evaluation"]
            self.pipes[i][0].close()
            self.workers[i].join()
