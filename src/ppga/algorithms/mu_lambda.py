import psutil
from tqdm import tqdm

from ppga import log
from ppga.algorithms.reproduction import reproduction
from ppga.algorithms.worker import Worker
from ppga.base import HallOfFame, Statistics, ToolBox


def mu_lambda(
    toolbox: ToolBox,
    population_size: int,
    mu: int,
    lam: int,
    cxpb: float,
    mutpb: float,
    max_generations: int,
    hall_of_fame: None | HallOfFame = None,
    log_level: str | int = log.WARNING,
):
    stats = Statistics()
    logger = log.getCoreLogger(log_level)

    population = toolbox.generate(population_size)
    for g in tqdm(range(max_generations), desc="generations", ncols=80):
        chosen = toolbox.select(population, mu)
        logger.debug(f"chosen: {len(chosen)}")

        offsprings = reproduction(chosen, toolbox, lam, cxpb, mutpb)
        logger.debug(f"offsprings generated: {len(offsprings)}")

        offsprings = list(toolbox.map(toolbox.evaluate, offsprings))

        stats.update_evals(len(offsprings))

        population = toolbox.replace(population, offsprings)
        logger.debug(f"population size: {len(population)}")

        stats.update(population)

        if hall_of_fame is not None:
            hall_of_fame.update(population)

    return population, stats


def parallel_mu_lambda(
    toolbox: ToolBox,
    population_size: int,
    mu: int,
    lam: int,
    cxpb: float,
    mutpb: float,
    max_generations: int,
    hall_of_fame: None | HallOfFame = None,
    log_level: str | int = log.WARNING,
):
    stats = Statistics()

    logger = log.getCoreLogger(log_level)

    # only use the physical cores
    workers_num = psutil.cpu_count(logical=False)

    # dinamically resize the chunksize
    chunksize = mu // workers_num
    carry = mu % workers_num
    logger.info(f"chunksize: {chunksize}")
    logger.info(f"carry: {carry}")

    workers = [Worker(toolbox, lam, cxpb, mutpb) for _ in range(workers_num)]

    population = toolbox.generate(population_size)
    for g in tqdm(range(max_generations), desc="generations", ncols=80):
        chosen = toolbox.select(population, mu)

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
