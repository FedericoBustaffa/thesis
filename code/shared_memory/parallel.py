import multiprocessing as mp
import multiprocessing.shared_memory as sm
import multiprocessing.sharedctypes as st
import multiprocessing.synchronize as sync
from multiprocessing.queues import Queue
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
    sizes: list,
    ready: sync.Event,
    ready_counter: st.Synchronized,
):
    while not ready.is_set():
        ready.wait()

    # print(f"{mp.current_process().name}")
    # couples_memory = sm.SharedMemory(name="couples_mem")
    # couples = np.ndarray(
    #     shape=info_queue.get_nowait(),
    #     dtype=np.int64,
    #     buffer=couples_memory.buf,
    # )

    # couples_memory.close()
