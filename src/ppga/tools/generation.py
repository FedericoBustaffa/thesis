import random


def repetition(values, length: int):
    return random.choices(values, k=length)


def permutation(values, length: int):
    return random.sample(values, k=length)
