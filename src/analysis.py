import os

import matplotlib.pyplot as plt
import numpy as np

import tsp


def main():
    tsp.main(["tsp.py", "10", "10000", "50", "info"])

    main_file = open("results/MainProcess.txt", "r")
    stats = []
    lines = main_file.readlines()
    for line in lines:
        stats.append(float(line))

    stats = np.array(stats)
    print(f"main total time: {stats.sum()} seconds")
    print(f"main mean time: {stats.mean()} seconds")

    files = [
        open(f"./results/{filepath}", "r")
        for filepath in os.listdir("./results/")
        if filepath.startswith("P") and filepath.endswith(".txt")
    ]

    pstats = []
    for f in files:
        lines = f.readlines()
        pstats.append(np.array([float(line) for line in lines]))

    totals = [s.sum() for s in pstats]
    means = [s.mean() for s in pstats]

    for i, (t, m) in enumerate(zip(totals, means)):
        print(f"worker {i+1} total time: {t}\t | mean time: {m} seconds")

    workers_num = len(pstats)

    plt.figure(figsize=(16, 9))
    plt.title("Total time per worker")
    plt.xlabel("Worker")
    plt.ylabel("Total time of work")
    plt.xticks([i + 1 for i in range(workers_num)])
    plt.bar([i + 1 for i in range(workers_num)], totals)
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.title("Mean time per worker")
    plt.xlabel("Worker")
    plt.ylabel("Mean time of work")
    plt.xticks([i for i in range(workers_num + 1)])
    plt.bar([0], stats.mean())
    plt.bar([i + 1 for i in range(workers_num)], means)
    plt.show()


if __name__ == "__main__":
    main()
