import sys
import time

from numpy import random

from ppga import algorithms, base, log, tools


def evaluate(chromosome) -> tuple:
    v = 0
    for _ in range(len(chromosome)):
        for _ in range(5000):
            v += random.random()

    return (v,)


def prepare_toolbox() -> base.ToolBox:
    toolbox = base.ToolBox()
    toolbox.set_weights(weights=(1.0,))
    toolbox.set_generation(tools.gen_repetition, (0, 1), 10)
    toolbox.set_selection(tools.sel_ranking)
    toolbox.set_crossover(tools.cx_uniform)
    toolbox.set_mutation(tools.mut_bitflip)
    toolbox.set_evaluation(evaluate)
    toolbox.set_replacement(tools.elitist, keep=0.3)

    return toolbox


def main(argv: list[str]) -> None:
    logger = log.getUserLogger()
    logger.setLevel("BENCHMARK")

    population_size = int(argv[1])
    generations = int(argv[2])

    toolbox = prepare_toolbox()

    start = time.perf_counter()
    best, stats = algorithms.elitist(
        toolbox=toolbox,
        population_size=population_size,
        keep=0.1,
        cxpb=0.8,
        mutpb=0.2,
        max_generations=generations,
    )
    stime = time.perf_counter() - start
    logger.log(15, f"sequential time: {stime} seconds")

    start = time.perf_counter()
    pbest, pstats = algorithms.pelitist(
        toolbox=toolbox,
        population_size=population_size,
        keep=0.1,
        cxpb=0.8,
        mutpb=0.2,
        max_generations=generations,
    )
    ptime = time.perf_counter() - start
    logger.log(15, f"sequential time: {ptime} seconds")

    logger.log(15, f"speed up: {stime / ptime}")


if __name__ == "__main__":
    main(sys.argv)
