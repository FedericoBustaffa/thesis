import pandas as pd
from common import Item, show_solution

from ppga import log


def greedy(items: list[Item], capacity: float) -> tuple[list[int], list[Item]]:
    items = sorted(items, key=lambda x: x.value / x.weight, reverse=True)
    weight = 0.0

    solution = []
    for i in items:
        if i.weight + weight <= capacity:
            solution.append(1)
            weight += i.weight
        else:
            solution.append(0)

    return solution, items


if __name__ == "__main__":
    nitems = [25, 50, 100, 200, 400]

    results = {"items": [], "capacity": [], "value": [], "weight": []}

    for n in nitems:
        df = pd.read_csv(f"problems/knapsack/datasets/items_{n}.csv")
        values = df["value"].to_list()
        weights = df["weight"].to_list()
        items = [Item(v, w) for v, w in zip(values, weights)]
        capacity = sum([i.weight for i in items]) * 0.5

        logger = log.getUserLogger()
        logger.setLevel("INFO")
        logger.info(f"capacity: {capacity:.3f}")

        solution, items = greedy(items, capacity)
        value, weight = show_solution(solution, items)
        logger.info(f"greedy (value: {value:.3f}, weight: {weight:.3f})")

        results["items"].append(n)
        results["capacity"].append(capacity)
        results["value"].append(value)
        results["weight"].append(weight)

    df = pd.DataFrame(results)
    df.to_csv("problems/knapsack/results/greedy.csv", index=False, header=True)
    print(df)
