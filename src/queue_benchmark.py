import multiprocessing as mp
import multiprocessing.queues as mpq
import time

import matplotlib.pyplot as plt
import numpy as np

from ppga import base


def benchmark(dim: int):
    q = mp.Queue()
    pop = [base.Individual([i for i in range(dim)])]
    put = []
    get = []

    for _ in range(10000):
        put_start = time.perf_counter()
        q.put(pop)
        get_start = time.perf_counter()
        pop = q.get()
        end = time.perf_counter()

        put.append(get_start - put_start)
        get.append(end - get_start)

    return np.mean(put), np.mean(get)


def main():
    pop_dims = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    put_means = []
    get_means = []

    for d in pop_dims:
        put, get = benchmark(d)
        put_means.append(put)
        get_means.append(get)

    plt.figure(figsize=(16, 9))
    plt.title("Mean times")
    plt.xticks(pop_dims)
    plt.xlabel("Population dimension")
    plt.ylabel("Time (s)")

    plt.plot(pop_dims, put_means, label="put")
    plt.plot(pop_dims, get_means, label="get")

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
