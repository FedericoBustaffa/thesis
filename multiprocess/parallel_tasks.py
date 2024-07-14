import os
import multiprocessing as mp
import multiprocessing.shared_memory as sm
import numpy as np
import time


def matrix_mult(a, b):
    return a @ b


if __name__ == "__main__":
    print("Dev branch")

    n = 4
    shape = (n, n)
    a = np.random.randint(1, 4, shape)
    b = np.random.randint(1, 4, shape)
    k = 4

    # Sequential
    start = time.perf_counter()
    for _ in range(k):
        c = matrix_mult(a, b)
    end = time.perf_counter()
    seq_time = end - start
    print(f"Sequential time: {seq_time} s")

    # Parallel
    workers = [mp.Process(target=matrix_mult, args=[a, b]) for _ in range(k)]
    start = time.perf_counter()
    for w in workers:
        w.start()

    for w in workers:
        w.join()
    end = time.perf_counter()
    parallel = end - start
    print(f"Parallel time: {parallel} s")

    print(f"Speed Up factor: {seq_time / parallel}")

    shared = sm.SharedMemory("Mem 1", True, 256)
    print(shared)
