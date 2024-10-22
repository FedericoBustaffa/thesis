import os
import time

from tqdm import tqdm

from ppga.algorithms.worker import Worker
from ppga.base.hall_of_fame import HallOfFame
from ppga.base.statistics import Statistics
from ppga.base.toolbox import ToolBox


def generational(
    toolbox: ToolBox,
    population_size: int,
    max_generations: int,
    hall_of_fame: None | HallOfFame = None,
):
    stats = Statistics()

    # start the parallel workers
    workers_num = os.cpu_count()
    assert workers_num is not None

    workers = [Worker(toolbox, stats) for _ in range(workers_num)]

    start = time.perf_counter()
    population = toolbox.generate(population_size)
    stats.add_time("generation", start)

    for g in tqdm(range(max_generations), desc="generations", ncols=80):
        start = time.perf_counter()
        chosen = toolbox.select(population, population_size)
        stats.add_time("selection", start)

        start = time.perf_counter()
        couples = toolbox.mate(chosen)
        stats.add_time("mating", start)

        # parallel crossover + mutation + evaluation
        chunksize = len(couples) // workers_num
        carry = len(couples) % workers_num

        start = time.perf_counter()
        for i in range(carry):
            workers[i].send(couples[i * chunksize : i * chunksize + chunksize + 1])

        for i in range(carry, workers_num, 1):
            workers[i].send(couples[i * chunksize : i * chunksize + chunksize])

        # keep only the worst time for each worker
        offsprings = []
        crossover_time = 0.0
        mutation_time = 0.0
        evaluation_time = 0.0

        for worker in workers:
            offsprings_chunk, timings = worker.recv()

            offsprings.extend(offsprings_chunk)
            s = sum([timings["crossover"], timings["mutation"], timings["evaluation"]])
            s2 = sum([crossover_time, mutation_time, evaluation_time])
            if s > s2:
                crossover_time = timings["crossover"]
                mutation_time = timings["mutation"]
                evaluation_time = timings["evaluation"]

        stats["crossover"] += crossover_time
        stats["mutation"] += mutation_time
        stats["evaluation"] += evaluation_time
        stats.add_time("parallel", start)

        # replacement
        start = time.perf_counter()
        population = toolbox.replace(population, offsprings)
        stats.add_time("replacement", start)

        stats.push_best(max(population).fitness)
        stats.push_worst(min([i for i in population if not i.invalid()]).fitness)

        if hall_of_fame is not None:
            hall_of_fame.update(population)

    for h in workers:
        h.join()

    return population, stats
