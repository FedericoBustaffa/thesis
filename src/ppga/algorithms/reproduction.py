import random

from ppga.base import Individual, ToolBox
from ppga.log import core_logger


def reproduction(
    chosen: list[Individual], toolbox: ToolBox, lam: int, cxpb: float, mutpb: float
):
    offsprings = []
    for i in range(0, lam // 2, 1):
        offspring1, offspring2 = random.choices(chosen, k=2)
        if random.random() <= cxpb:
            offspring1, offspring2 = toolbox.crossover(offspring1, offspring2)

        if random.random() <= mutpb:
            offspring1 = toolbox.mutate(offspring1)

        if random.random() <= mutpb:
            offspring2 = toolbox.mutate(offspring2)

        offsprings.extend([offspring1, offspring2])

    core_logger.debug(f"lambda: {lam}")

    return offsprings
