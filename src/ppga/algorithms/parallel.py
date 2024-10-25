import os

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

    workers = [Worker(toolbox) for _ in range(workers_num)]

    population = toolbox.generate(population_size)

    for g in tqdm(range(max_generations), desc="generations", ncols=80):
        chosen = toolbox.select(population, population_size)
        couples = toolbox.mate(chosen)

        # parallel crossover + mutation + evaluation
        chunksize = len(couples) // workers_num
        carry = len(couples) % workers_num

        for i in range(carry):
            workers[i].send(couples[i * chunksize : i * chunksize + chunksize + 1])

        for i in range(carry, workers_num, 1):
            workers[i].send(couples[i * chunksize : i * chunksize + chunksize])

        # keep only the worst time for each worker
        offsprings = []
        for worker in workers:
            offsprings_chunk = worker.recv()
            offsprings.extend(offsprings_chunk)

        # replacement
        population = toolbox.replace(population, offsprings)
        stats.update(population)

        if hall_of_fame is not None:
            hall_of_fame.update(population)

    for h in workers:
        h.join()

    return population, stats
