import sys

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(f"USAGE: py {sys.argv[0]} <stats_filename>")
        exit(1)

    df = pd.read_csv(f"stats/{sys.argv[1]}")
    mask = (
        (df["cities"] == 100)
        & (df["population_size"] == 2000)
        & (df["generations"] == 500)
    )
    df = df[mask]

    # calculate the sequential time mean
    mask = df["implementation"] == "sequential"
    seq_mean_time = df[mask]["time"].mean()

    # calculate the pipe time mean
    mask = (df["implementation"] == "pipe") | (df["implementation"] == "shared memory")
    df = df[mask]
    df = df.groupby(["implementation", "workers"])["time"].mean()

    # plot the ideal time
    workers = [i + 1 for i in range(20)]

    plt.title("Speed up")
    plt.xlabel("Number of workers")
    plt.ylabel("Speed Up")

    plt.plot(workers, workers, label="ideal")
    plt.plot(df["pipe"].index, seq_mean_time / df["pipe"].values, label="pipe")
    plt.plot(
        df["shared memory"].index,
        seq_mean_time / df["shared memory"].values,
        label="shared memory",
    )

    plt.grid()
    plt.legend()
    plt.show()
