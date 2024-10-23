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

    times = []

    for g in tqdm(range(max_generations), desc="generations", ncols=80):
        start = time.perf_counter()
        chosen = toolbox.select(population, population_size)
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
        offsprings, mean_time = toolbox.evaluate(offsprings)
        stats.add_time("evaluation", start)
        times.append(mean_time)

        start = time.perf_counter()
        population = toolbox.replace(population, offsprings)
        stats.add_time("replacement", start)

        stats.push_best(max(population).fitness)
        stats.push_worst(min([i for i in population if not i.invalid()]).fitness)

        if hall_of_fame is not None:
            hall_of_fame.update(population)

    print(f"eval mean time: {sum(times) / len(times)} seconds")

    return population, stats
