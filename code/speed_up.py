import sys

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(
            f"USAGE: py {sys.argv[0]} <stats_filename> <cities> <population_size> <generations> <workers>"
        )
        exit(1)

    filepath = sys.argv[1]
    cities = int(sys.argv[2])
    population_size = int(sys.argv[3])
    generations = int(sys.argv[4])
    workers = int(sys.argv[5])

    df = pd.read_csv(filepath)
    mask = (
        (df["cities"] == cities)
        & (df["population_size"] == population_size)
        & (df["generations"] == generations)
    )
    df = df[mask]

    print(df)

    # calculate the sequential time mean
    mask = df["implementation"] == "sequential"
    seq_mean_time = df.where(mask)["time"].mean()

    # calculate the pipe time mean
    mask = (df["implementation"] == "pipe") | (df["implementation"] == "shared memory")
    df = df.where(mask)
    df = df.groupby(["implementation", "workers"])["time"].mean()

    print(df)

    # plot the ideal time
    workers = [i + 1 for i in range(workers)]

    plt.title("Speed up")
    plt.xlabel("Number of workers")
    plt.ylabel("Speed Up")
    plt.xticks(workers)

    plt.plot(workers, workers, label="ideal")

    plt.plot(
        df.loc["pipe"].index,
        seq_mean_time / df.loc["pipe"].values,
        label="pipe",
    )

    plt.plot(
        df.loc["shared memory"].index,
        seq_mean_time / df.loc["shared memory"].values,
        label="shared memory",
    )

    plt.grid()
    plt.legend()
    plt.show()
