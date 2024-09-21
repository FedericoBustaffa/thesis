import multiprocessing as mp
import multiprocessing.shared_memory as sm
import multiprocessing.sharedctypes as st
import multiprocessing.synchronize as sync
import time
from multiprocessing.connection import Connection

import numpy as np


def share(buffer, mem_name):

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
    num_of_workers: int,
    pipe: Connection,
    ready: sync.Event,
    ready_counter: st.Synchronized,
):
    shape, dtype = pipe.recv()

    # print(f"{mp.current_process().name}")
    couples_memory = sm.SharedMemory(name="couples_mem")
    couples = np.ndarray(
        shape=shape,
        dtype=dtype,
        buffer=couples_memory.buf,
    )

    chunk = len(couples) // num_of_workers
    for i in range(index * chunk, index * chunk + chunk, 1):
        print(f"{mp.current_process().name}: {couples[i]}")

    print(ready.is_set())
    while not ready.wait():
        pass

    print("work in progress")
    time.sleep(2)
    with ready_counter:
        ready_counter.value -= 1
        ready.set()

    couples_memory.close()
    pipe.close()
