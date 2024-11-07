import psutil

from ppga import log, tools
from ppga.algorithms.reproduction import reproduction
from ppga.algorithms.worker import Worker
from ppga.base import HallOfFame, Statistics, ToolBox


def mu_lambda(
    toolbox: ToolBox,
    population_size: int,
    mu: int,
    keep: float,
    cxpb: float,
    mutpb: float,
    max_generations: int,
    hall_of_fame: None | HallOfFame = None,
    log_level: str | int = log.WARNING,
):
    """
    Simple genetic algorithm that select `mu` individuals every generation
    for reproduction and try to keep `lam` individuals to the new generation.
    """

    assert keep >= 0.0 and keep <= 1.0
    toolbox.set_replacement(tools.partial, keep=keep)

    stats = Statistics()
    logger = log.getCoreLogger(log_level)

    population = toolbox.generate(population_size)
    logger.info(f"\t{'gen':15s}{'evals':15s}")

    for g in range(max_generations):
        chosen = toolbox.select(population, mu)
        offsprings = reproduction(chosen, toolbox, cxpb, mutpb)
        offsprings = list(map(toolbox.evaluate, offsprings))
        population = toolbox.replace(population, offsprings)

        if hall_of_fame is not None:
            hall_of_fame.update(population)

        stats.update(population)
        stats.update_evals(len(offsprings))

        logger.info(f"\t{g:<15d}{len(offsprings):<15d}")

    return population, stats


def parallel_mu_lambda(
    toolbox: ToolBox,
    population_size: int,
    mu: int,
    keep: float,
    cxpb: float,
    mutpb: float,
    max_generations: int,
    hall_of_fame: None | HallOfFame = None,
    log_level: str | int = log.WARNING,
):
    assert keep >= 0.0 and keep <= 1.0
    toolbox.set_replacement(tools.partial, keep=keep)

    stats = Statistics()

    logger = log.getCoreLogger(log_level)

    # only use the physical cores
    workers_num = psutil.cpu_count(logical=False)

    # dinamically resize the chunksize
    if mu < workers_num:
        workers_num = mu

    chunksize = mu // workers_num
    carry = mu % workers_num
    logger.debug(f"chunksize: {chunksize}")
    logger.debug(f"carry: {carry}")
    logger.debug(f"workers: {workers_num}")

    workers = [Worker(toolbox, cxpb, mutpb, log_level) for _ in range(workers_num)]

    population = toolbox.generate(population_size)
    logger.info(f"\t{'gen':15s}{'evals':15s}")
    for g in range(max_generations):
        chosen = toolbox.select(population, mu)
        chosen = [toolbox.clone(i) for i in chosen]

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

        logger.info(f"\t{g:<15d}{len(offsprings):<15d}")

        if hall_of_fame is not None:
            hall_of_fame.update(population)

    for w in workers:
        w.join()

    return population, stats
