import math
import multiprocessing as mp

from generation import generate
from selection import select
from crossover import mating
from parallel import work
from replacement import replace


class PipeGeneticAlgorithm:

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
        self.offsprings = []

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

    generate = generate
    select = select
    mating = mating
    work = work
    replace = replace

    def run(self, max_generations: int):

        # generate random population
        self.generate()
        # print("generated")
        # for i in self.population:
        #     print(i)

        self.best = self.population[0]
        print(f"first best: {self.best.fitness}")

        for g in range(max_generations):
            # print(g)
            # select individuals for crossover
            self.select()
            # print("selected")
            # for i in self.selected:
            #     print(f"{i}. {self.population[i]}")

            # creating couples for crossover
            couples = self.mating()
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
