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

    population = toolbox.generate(population_size)
    for g in tqdm(range(max_generations), desc="generations", ncols=80):
        chosen = toolbox.select(population, population_size)
        couples = toolbox.mate(chosen)
        offsprings = toolbox.crossover(couples)
        offsprings = toolbox.mutate(offsprings)
        offsprings = toolbox.evaluate(offsprings)
        population = toolbox.replace(population, offsprings)

        stats.update(population)

        if hall_of_fame is not None:
            hall_of_fame.update(population)

    return population, stats
