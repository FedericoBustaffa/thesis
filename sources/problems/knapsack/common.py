import math


class Item:
    def __init__(self, value: float, weight: float):
        self.value = value
        self.weight = weight


def evaluate(chromosome, items: list[Item], capacity: float) -> tuple:
    value = 0.0
    weight = 0.0

    for i in range(len(items)):
        value += chromosome[i] * items[i].value
        weight += chromosome[i] * items[i].weight

    if weight > capacity:
        return -math.inf, math.inf
    else:
        return value, weight


def show_solution(solution, items):
    value = sum([i.value * s for i, s in zip(items, solution)])
    weight = sum([i.weight * s for i, s in zip(items, solution)])

    return value, weight
