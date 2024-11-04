import os
import random

from tqdm import tqdm

from ppga import log
from ppga.algorithms.worker import Worker
from ppga.base import HallOfFame, Individual, Statistics, ToolBox


def reproduction(chosen: list[Individual], toolbox: ToolBox, cxpb: float, mutpb: float):
    offsprings = []
    for i in range(0, len(chosen), 1):
        if random.random() <= cxpb:
            father, mother = random.choices(chosen, k=2)
            offspring1, offspring2 = toolbox.crossover(father, mother)

            if random.random() <= mutpb:
                offspring1 = toolbox.mutate(offspring1)

            if random.random() <= mutpb:
                offspring2 = toolbox.mutate(offspring2)

            offsprings.extend([offspring1, offspring2])

    return offsprings


def sga(
    toolbox: ToolBox,
    population_size: int,
    cxpb: float,
    mutpb: float,
    max_generations: int,
    hall_of_fame: None | HallOfFame = None,
):
    stats = Statistics()

    population = toolbox.generate(population_size)
    for g in tqdm(range(max_generations), desc="generations", ncols=80):
        chosen = toolbox.select(population, population_size)
        offsprings = reproduction(chosen, toolbox, cxpb, mutpb)

        for offspring in offsprings:
            offspring = toolbox.evaluate(offspring)

        stats.update_evals(len(offsprings))

        population = toolbox.replace(population, offsprings)

        stats.update(population)

        if hall_of_fame is not None:
            hall_of_fame.update(population)

    return population, stats


def psga(
    toolbox: ToolBox,
    population_size: int,
    cxpb: float,
    mutpb: float,
    max_generations: int,
    hall_of_fame: None | HallOfFame = None,
):
    stats = Statistics()

    workers_num = 32
    assert workers_num is not None

    workers = [Worker(toolbox, cxpb, mutpb) for _ in range(workers_num)]

    population = toolbox.generate(population_size)
    for g in tqdm(range(max_generations), desc="generations", ncols=80):
        chosen = toolbox.select(population, population_size)
        chunksize = len(chosen) // workers_num
        carry = len(chosen) % workers_num

        for i in range(carry):
            workers[i].send(chosen[i * chunksize : i * chunksize + chunksize + 1])

        for i in range(carry, workers_num, 1):
            workers[i].send(chosen[i * chunksize : i * chunksize + chunksize])

        # keep only the worst time for each worker
        offsprings = []
        evals = []
        for worker in workers:
            offsprings_chunk = worker.recv()
            offsprings.extend(offsprings_chunk)
            evals.append(len(offsprings_chunk))

        population = toolbox.replace(population, offsprings)

        stats.update(population)
        stats.update_multievals(evals)

        if hall_of_fame is not None:
            hall_of_fame.update(population)

    for w in workers:
        w.join()

    return population, stats
