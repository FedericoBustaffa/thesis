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
    population = toolbox.generate(population_size)

    logger.info(f"\t{'gen':15s}{'evals':15s}")

    for g in range(max_generations):
        # select individuals for reproduction
        chosen = toolbox.select(population, population_size)

        # perform crossover and mutation
        offsprings = reproduction(chosen, toolbox, cxpb, mutpb)

        # evaluate the individuals with invalid fitness
        for i in range(len(offsprings)):
            if offsprings[i].invalid:
                offsprings[i] = toolbox.evaluate(offsprings[i])

        # elitist replacement
        population = toolbox.replace(population, offsprings)

        # update the Hall of Fame if present
        if hall_of_fame is not None:
            hall_of_fame.update(population)

        # update the stats
        stats.update(population)
        stats.update_evals(len([i for i in offsprings if i.invalid]))

        logger.info(f"\t{g:<15d}{len(offsprings):<15d}")

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

    population = toolbox.generate(population_size)

    logger.info(f"\t{'gen':15s}{'evals':15s}")

    for g in range(max_generations):
        chosen = toolbox.select(population, population_size)

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

        population = toolbox.replace(population, offsprings)

        stats.update(population)
        stats.update_multievals(evals)

        logger.info(f"\t{g:<15d}{stats.evals[-1]:<15d}")

        if hall_of_fame is not None:
            hall_of_fame.update(population)

    for w in workers:
        w.join()

    return population, stats
