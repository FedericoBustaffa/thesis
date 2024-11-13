import multiprocessing as mp
import multiprocessing.queues as mpq
import multiprocessing.synchronize as sync
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np

from ppga import base


def queue_receive(event: sync.Event, queue: mpq.Queue, ret: mpq.Queue) -> None:
    times = []
    event.wait()
    while True:
        start = time.perf_counter()
        msg = queue.get()
        end = time.perf_counter()

        if msg is None:
            break

        times.append(end - start)

    ret.put(times)


def queue_benchmark(dim: int):
    q = mp.Queue()
    ret = mp.Queue()
    e = mp.Event()
    p = mp.Process(target=queue_receive, args=[e, q, ret])
    p.start()

    pop = [base.Individual([i for i in range(dim)])]
    put = []

    for _ in range(1000):
        start = time.perf_counter()
        q.put(pop)
        end = time.perf_counter()
        put.append(end - start)

    e.set()
    q.put(None)
    get = np.array(ret.get())

    q.close()
    q.join_thread()
    p.join()

    return np.mean(put), get.mean()


def main():
    pop_dims = [
        100,
        200,
        500,
        1000,
        2000,
        5000,
        10000,
        20000,
        50000,
        100000,
        200000,
        500000,
    ]
    put_means = []
    get_means = []

    for d in pop_dims:
        print(d)
        put, get = queue_benchmark(d)
        put_means.append(put)
        get_means.append(get)

    reg = statistics.linear_regression(pop_dims, get_means)
    print(f"get slope: {reg.slope}")

    reg = statistics.linear_regression(pop_dims, put_means)
    print(f"put slope: {reg.slope}")

    plt.figure(figsize=(16, 9))
    plt.title("Mean times")
    plt.xlabel("Population dimension")
    plt.ylabel("Time (s)")

    plt.plot(pop_dims, put_means, label="put")
    plt.plot(pop_dims, get_means, label="get")

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
