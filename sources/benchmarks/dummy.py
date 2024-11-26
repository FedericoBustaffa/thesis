from numpy import random

from ppga import base, tools


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
