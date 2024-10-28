import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    files = [
        open(filepath, "r")
        for filepath in os.listdir(os.getcwd())
        if filepath.startswith("T") and filepath.endswith(".txt")
    ]

    stats = []
    for f in files:
        lines = f.readlines()
        stats.append(np.array([float(line) for line in lines]))

    stats = np.array(stats).T

    total_time = stats.sum(axis=0)
    print(f"max time: {total_time.max()}")
    print(f"min time: {total_time.min()}")
    mean_time = stats.mean(axis=0)

    plt.figure(figsize=(16, 9))
    plt.title("Total time per worker")
    plt.xlabel("Worker")
    plt.ylabel("Total time of work")
    plt.xticks([i + 1 for i in range(len(total_time))])
    plt.bar([i + 1 for i in range(len(total_time))], total_time)
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.title("Mean time per worker")
    plt.xlabel("Worker")
    plt.ylabel("Mean time of work")
    plt.xticks([i + 1 for i in range(len(total_time))])
    plt.bar([i + 1 for i in range(len(mean_time))], mean_time)
    plt.show()


if __name__ == "__main__":
    main()
