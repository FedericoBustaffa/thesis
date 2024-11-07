import random

from ppga.base import Individual, ToolBox


def reproduction(chosen: list[Individual], toolbox: ToolBox, cxpb: float, mutpb: float):
    offsprings = []
    chosen = [toolbox.clone(i) for i in chosen]
    for i in range(0, len(chosen), 2):
        if random.random() <= cxpb:
            father, mother = random.choices(chosen, k=2)
            offspring1, offspring2 = toolbox.crossover(father, mother)

            if random.random() <= mutpb:
                offspring1 = toolbox.mutate(offspring1)

            if random.random() <= mutpb:
                offspring2 = toolbox.mutate(offspring2)

            offsprings.extend([offspring1, offspring2])

    return offsprings
