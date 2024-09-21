import multiprocessing as mp
import multiprocessing.shared_memory as sm
import multiprocessing.sharedctypes as st
import multiprocessing.synchronize as sync
import random
import time
from multiprocessing.connection import Connection

import numpy as np


def share(self, buffer, mem_name):

    buffer = np.array(buffer)
    buffer_memory = sm.SharedMemory(name=mem_name, create=True, size=buffer.nbytes)

    shared_buffer = np.ndarray(
        shape=buffer.shape,
        dtype=buffer.dtype,
        buffer=buffer_memory.buf,
    )
    shared_buffer[:] = buffer[:]

    return buffer_memory, shared_buffer


def parallel_work(
    self,
    index: int,
    workers_num: int,
    pipe: Connection,
    main_ready: sync.Event,
    ready: sync.Event,
    stop: st.Synchronized,
):
    shape, dtype = pipe.recv()
    couples_memory = sm.SharedMemory(name="couples_mem", create=False)
    couples = np.ndarray(
        shape=shape,
        dtype=dtype,
        buffer=couples_memory.buf,
    )

    population_memory = sm.SharedMemory(name="population_mem", create=False)
    population = np.ndarray(
        shape=shape,
        dtype=dtype,
        buffer=couples_memory.buf,
    )

    scores_memory = sm.SharedMemory(name="scores_mem", create=False)
    scores = np.ndarray(
        shape=shape,
        dtype=dtype,
        buffer=couples_memory.buf,
    )

    chunk = len(couples) // workers_num

    while True:

        main_ready.wait()
        main_ready.clear()

        with stop:
            if stop.value == 1:
                break

        for i in range(index * chunk, index * chunk + chunk, 1):
            father = population[couples[i][0]]
            mother = population[couples[i][1]]

        ready.set()

    couples_memory.close()
    pipe.close()
