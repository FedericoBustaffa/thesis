import time

import psutil

from ppga import log, tools
from ppga.algorithms.reproduction import reproduction
from ppga.algorithms.worker import Worker
from ppga.base import HallOfFame, Statistics, ToolBox


def generational(
    toolbox: ToolBox,
    population_size: int,
    cxpb: float,
    mutpb: float,
    max_generations: int,
    hall_of_fame: None | HallOfFame = None,
    log_level: str | int = log.WARNING,
):
    toolbox.set_replacement(tools.total)

    stats = Statistics()
    logger = log.getCoreLogger(log_level)

    # generate the initial population
    start = time.perf_counter()
    population = toolbox.generate(population_size)
    generation_time = time.perf_counter() - start

    logger.info(f"\t{'gen':15s}{'evals':15s}")

    selection_time = 0.0
    parallel_time = 0.0
    replace_time = 0.0

    for g in range(max_generations):
        # select individuals for reproduction
        start = time.perf_counter()
        chosen = toolbox.select(population, population_size)
        selection_time += time.perf_counter() - start

        # perform crossover and mutation
        start = time.perf_counter()
        offsprings = reproduction(chosen, toolbox, cxpb, mutpb)

        # evaluate the individuals with invalid fitness
        invalid_individuals = [i for i in offsprings if i.invalid]
        offsprings = list(map(toolbox.evaluate, invalid_individuals))
        parallel_time += time.perf_counter() - start

        # elitist replacement
        start = time.perf_counter()
        population = toolbox.replace(population, offsprings)
        replace_time += time.perf_counter() - start

        # update the Hall of Fame if present
        if hall_of_fame is not None:
            hall_of_fame.update(population)

        # update the stats
        stats.update(population)
        stats.update_evals(len(invalid_individuals))

        logger.info(f"\t{g:<15d}{len(offsprings):<15d}")

    logger.info(f"generation time: {generation_time:.4f} seconds")
    logger.info(f"selection time: {selection_time:.4f} seconds")
    logger.info(f"parallel time: {parallel_time:.4f} seconds")
    logger.info(f"replacement time: {replace_time:.4f} seconds")

    return population, stats


def pgenerational(
    toolbox: ToolBox,
    population_size: int,
    cxpb: float,
    mutpb: float,
    max_generations: int,
    hall_of_fame: None | HallOfFame = None,
    log_level: str | int = log.WARNING,
):
    toolbox.set_replacement(tools.total)

    stats = Statistics()

    logger = log.getCoreLogger(log_level)

    # only use the physical cores
    workers_num = psutil.cpu_count(logical=False)

    # dinamically resize the chunksize
    if population_size < workers_num:
        workers_num = population_size

    chunksize = population_size // workers_num
    carry = population_size % workers_num

    workers = [Worker(toolbox, cxpb, mutpb, log_level) for _ in range(workers_num)]

    start = time.perf_counter()
    population = toolbox.generate(population_size)
    generation_time = time.perf_counter() - start

    selection_time = 0.0
    parallel_time = 0.0
    replace_time = 0.0

    logger.info(f"\t{'gen':15s}{'evals':15s}")

    for g in range(max_generations):
        start = time.perf_counter()
        chosen = toolbox.select(population, population_size)
        selection_time += time.perf_counter() - start

        start = time.perf_counter()
        for i in range(carry):
            workers[i].send(chosen[i * chunksize : i * chunksize + chunksize + 1])

        for i in range(carry, workers_num, 1):
            workers[i].send(chosen[i * chunksize : i * chunksize + chunksize])

        offsprings = []
        evals = []
        for worker in workers:
            offsprings_chunk = worker.recv()
            offsprings.extend(offsprings_chunk)
            evals.append(len(offsprings_chunk))
        parallel_time += time.perf_counter() - start

        start = time.perf_counter()
        population = toolbox.replace(population, offsprings)
        replace_time += time.perf_counter() - start

        stats.update(population)
        stats.update_multievals(evals)

        logger.info(f"\t{g:<15d}{len(offsprings):<15d}")

        if hall_of_fame is not None:
            hall_of_fame.update(population)

    for w in workers:
        w.join()

    logger.info(f"generation time: {generation_time:.4f} seconds")
    logger.info(f"selection time: {selection_time:.4f} seconds")
    logger.info(f"parallel time: {parallel_time:.4f} seconds")
    logger.info(f"replacement time: {replace_time:.4f} seconds")

    return population, stats
