import random
import sys

import pandas as pd

if __name__ == "__main__":
    n = int(sys.argv[1])

    values = [random.random() for _ in range(n)]
    weights = [random.random() for _ in range(n)]

    df = pd.DataFrame({"value": values, "weight": weights})
    df.to_csv(f"problems/knapsack/datasets/items_{n}.csv", header=True, index=False)
