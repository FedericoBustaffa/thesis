import time

from tqdm import tqdm

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

    start = time.perf_counter()
    population = toolbox.generate(population_size)
    stats.add_time("generation", start)

    # start = time.perf_counter()
    population = toolbox.evaluate(population)
    # stats.add_time("evaluation", start)

    for g in tqdm(range(max_generations), desc="generations", ncols=80):
        start = time.perf_counter()
        chosen = toolbox.select(population)
        stats.add_time("selection", start)

        start = time.perf_counter()
        couples = toolbox.mate(chosen)
        stats.add_time("mating", start)

        start = time.perf_counter()
        offsprings = toolbox.crossover(couples)
        stats.add_time("crossover", start)

        start = time.perf_counter()
        offsprings = toolbox.mutate(offsprings)
        stats.add_time("mutation", start)

        start = time.perf_counter()
        offsprings = toolbox.evaluate(offsprings)
        stats.add_time("evaluation", start)

        start = time.perf_counter()
        population = toolbox.replace(population, offsprings)
        stats.add_time("replacement", start)

        stats.push_best(population[0].fitness.fitness)
        stats.push_worst(population[-1].fitness.fitness)

        if hall_of_fame is not None:
            hall_of_fame.update(population)

    return population, stats
