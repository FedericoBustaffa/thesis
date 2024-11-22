import logging
import random
import sys
import time

import dummy

from ppga import base, log, parallel
from ppga.algorithms import batch


def ptask(couples: list, toolbox: base.ToolBox):
    logger = log.getUserLogger()

    # crossover
    offsprings = []
    for father, mother in couples:
        if random.random() <= 0.8:
            start = time.perf_counter()
            offspring1, offspring2 = toolbox.crossover(father, mother)
            offsprings.extend([toolbox.clone(offspring1), toolbox.clone(offspring2)])
            cx_time = time.perf_counter() - start
            logger.log(15, f"crossover: {cx_time} seconds")

    # mutation
    for i, ind in enumerate(offsprings):
        if random.random() <= 0.2:
            start = time.perf_counter()
            offsprings[i] = toolbox.mutate(ind)
            mut_time = time.perf_counter() - start
            logger.log(15, f"mutation: {mut_time} seconds")

    # evaluation
    for i, ind in enumerate(offsprings):
        start = time.perf_counter()
        offsprings[i] = toolbox.evaluate(ind)
        eval_time = time.perf_counter() - start
        logger.log(15, f"evaluation: {eval_time} seconds")

    return offsprings


def parallel_run(toolbox: base.ToolBox, pop_size: int, max_gens: int):
    logger = log.getUserLogger()

    pool = parallel.Pool()

    # generation
    start = time.perf_counter()
    population = toolbox.generate(pop_size)
    gen_time = time.perf_counter() - start
    logger.log(15, f"generation: {gen_time} seconds")

    for g in range(max_gens):
        # selection
        start = time.perf_counter()
        selected = toolbox.select(population, pop_size)
        sel_time = time.perf_counter() - start
        logger.log(15, f"selection: {sel_time} seconds")

        # mating
        couples = batch.mating(selected)

        # parallel crossover, mutation and evaluation
        start = time.perf_counter()
        offsprings = pool.map(ptask, couples, args=[toolbox])
        parallel_time = time.perf_counter() - start
        logger.log(15, f"parallel: {parallel_time} seconds")

        # replacement
        start = time.perf_counter()
        population = toolbox.replace(population, offsprings)
        replace_time = time.perf_counter() - start
        logger.log(15, f"replacement: {replace_time} seconds")

    pool.join()


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        print(f"USAGE: py {argv[0]} <N> <G>")
        exit(1)

    logger = log.getUserLogger()
    logger.setLevel(15)

    formatter = log.JsonFormatter()
    handler = logging.FileHandler("logs/parallel.json", mode="w")
    handler.setFormatter(formatter)
    handler.setLevel(15)
    logger.addHandler(handler)

    toolbox = dummy.prepare_toolbox()

    start = time.perf_counter()
    parallel_run(toolbox, int(argv[1]), int(argv[2]))
    ptime = time.perf_counter() - start
    logger.log(15, f"ptime: {ptime} seconds")

    handler.close()


if __name__ == "__main__":
    main(sys.argv)
