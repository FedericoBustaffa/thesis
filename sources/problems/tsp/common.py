import math


class Town:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.visited = False

    def __repr__(self) -> str:
        return f"{self.x}, {self.y}"


def distance(t1: Town, t2: Town) -> float:
    return math.sqrt(math.pow(t1.x - t2.x, 2) + math.pow(t1.y - t2.y, 2))


def evaluate(chromosome, towns: list[Town]) -> tuple[float]:
    total_distance = 0.0
    for i in range(len(chromosome) - 1):
        total_distance += distance(towns[chromosome[i]], towns[chromosome[i + 1]])

    return (total_distance,)
