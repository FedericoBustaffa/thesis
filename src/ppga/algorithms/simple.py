from ppga.algorithms.mu_lambda import mu_lambda, parallel_mu_lambda
from ppga.base import HallOfFame, ToolBox


def sga(
    toolbox: ToolBox,
    population_size: int,
    cxpb: float,
    mutpb: float,
    max_generations: int,
    hall_of_fame: None | HallOfFame = None,
):
    return mu_lambda(
        toolbox=toolbox,
        population_size=population_size,
        mu=population_size,
        lam=population_size,
        cxpb=cxpb,
        mutpb=mutpb,
        max_generations=max_generations,
        hall_of_fame=hall_of_fame,
    )


def psga(
    toolbox: ToolBox,
    population_size: int,
    cxpb: float,
    mutpb: float,
    max_generations: int,
    hall_of_fame: None | HallOfFame = None,
):
    return parallel_mu_lambda(
        toolbox=toolbox,
        population_size=population_size,
        mu=population_size,
        lam=population_size,
        cxpb=cxpb,
        mutpb=mutpb,
        max_generations=max_generations,
        hall_of_fame=hall_of_fame,
    )
