import psutil

from ppga import log, tools
from ppga.algorithms.reproduction import reproduction
from ppga.algorithms.worker import Worker
from ppga.base import HallOfFame, Statistics, ToolBox


def elitist(
    toolbox: ToolBox,
    population_size: int,
    cxpb: float,
    mutpb: float,
    max_generations: int,
    hall_of_fame: None | HallOfFame = None,
    log_level: str | int = log.WARNING,
):
    toolbox.set_replacement(tools.elitist)

    stats = Statistics()
    logger = log.getCoreLogger(log_level)

    population = toolbox.generate(population_size)
    logger.info(f"\t{'gen':15s}{'evals':15s}")

    for g in range(max_generations):
        # select individuals for reproduction
        chosen = toolbox.select(population, population_size)

        # perform reproduction and mutation
        offsprings = reproduction(chosen, toolbox, cxpb, mutpb)

        # evaluate the individuals with invalid fitness
        evals = sum([1 for i in offsprings if not i.invalid])
        offsprings = list(map(toolbox.evaluate, offsprings))

        # elitist replacement
        population = toolbox.replace(population, offsprings)

        # update the Hall of Fame if present
        if hall_of_fame is not None:
            hall_of_fame.update(population)

        # update the stats
        stats.update(population)
        stats.update_evals(evals)

        logger.info(f"\t{g:<15d}{len(offsprings):<15d}")

    return population, stats


def pelitist(
    toolbox: ToolBox,
    population_size: int,
    cxpb: float,
    mutpb: float,
    max_generations: int,
    hall_of_fame: None | HallOfFame = None,
    log_level: str | int = log.WARNING,
):
    toolbox.set_replacement(tools.elitist)

    stats = Statistics()

    logger = log.getCoreLogger(log_level)

    # only use the physical cores
    workers_num = psutil.cpu_count(logical=False)

    # dinamically resize the chunksize
    if population_size < workers_num:
        workers_num = population_size

    chunksize = population_size // workers_num
    carry = population_size % workers_num
    logger.debug(f"chunksize: {chunksize}")
    logger.debug(f"carry: {carry}")
    logger.debug(f"workers: {workers_num}")

    workers = [Worker(toolbox, cxpb, mutpb, log_level) for _ in range(workers_num)]

    population = toolbox.generate(population_size)
    logger.info(f"\t{'gen':15s}{'evals':15s}")
    for g in range(max_generations):
        # select individuals for reproduction
        chosen = toolbox.select(population, population_size)

        # perform crossover, mutation and evaluation in parallel
        for i in range(carry):
            workers[i].send(chosen[i * chunksize : i * chunksize + chunksize + 1])

        for i in range(carry, workers_num, 1):
            workers[i].send(chosen[i * chunksize : i * chunksize + chunksize])

        # results of parallel computation
        offsprings = []
        evals = []
        for worker in workers:
            offsprings_chunk = worker.recv()
            offsprings.extend(offsprings_chunk)
            evals.append(len(offsprings_chunk))

        population = toolbox.replace(population, offsprings)

        # update stats
        stats.update(population)
        stats.update_multievals(evals)

        logger.info(f"\t{g:<15d}{len(offsprings):<15d}")

        # update Hall of Fame if present
        if hall_of_fame is not None:
            hall_of_fame.update(population)

    for w in workers:
        w.join()

    return population, stats
