import multiprocessing as mp
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np

from ppga import base


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
    ]

    recvs = []
    sends = []

    mean_recvs = []
    mean_sends = []

    local, remote = mp.Pipe()
    for i, d in enumerate(pop_dims):
        pop = [base.Individual([i for i in range(d)])]
        for _ in range(100):
            start = time.perf_counter()
            print(f"{i+1}: {d}")
            local.send(pop)
            send_end = time.perf_counter()
            print(remote.recv())
            recv_end = time.perf_counter()

            sends.append(send_end - start)
            recvs.append(recv_end - send_end)

        mean_recvs.append(np.mean(recvs))
        mean_sends.append(np.mean(sends))

        recvs.clear()
        sends.clear()

    local.close()
    remote.close()

    reg = statistics.linear_regression(pop_dims, mean_recvs)
    print(f"recv slope: {reg.slope}")

    reg = statistics.linear_regression(pop_dims, mean_sends)
    print(f"send slope: {reg.slope}")

    plt.figure(figsize=(16, 9))
    plt.title("Mean times")
    plt.xlabel("Population dimension")
    plt.ylabel("Time (s)")

    plt.plot(pop_dims, mean_recvs, label="recv")
    plt.plot(pop_dims, mean_sends, label="send")

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
