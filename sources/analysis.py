import argparse

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "population_size",
        type=int,
        help="specify the population size to analyze",
    )

    parser.add_argument(
        "workers",
        type=int,
        help="specify the number of worker to analyze",
    )

    args = parser.parse_args()

    df = pd.read_csv("results/ppga_MLPClassifier_32_multi.csv")

    df1 = df[
        (df["population_size"] == args.population_size)
        & (df["workers"] == args.workers)
    ]
    same_df = df1[df1["class"] == df1["target"]]
    diff_df = df1[df1["class"] != df1["target"]]

    plt.figure(figsize=(16, 9), dpi=150)
    plt.title(
        f"""Time comparison
        Population size: {args.population_size} - Workers: {args.workers}"""
    )
    plt.bar(
        same_df["point"].values - 0.06,
        same_df["time"],
        width=0.1,
        label="same class",
    )
    plt.bar(
        diff_df["point"].values + 0.06,
        diff_df["time"],
        width=0.1,
        label="different class",
    )
    plt.grid()
    plt.show()
