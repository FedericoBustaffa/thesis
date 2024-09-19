import multiprocessing as mp
import multiprocessing.shared_memory as sm
import multiprocessing.sharedctypes as st
import multiprocessing.synchronize as sync

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
    condition: sync.Condition,
    main_ready: st.Synchronized,
    process_ready: st.Synchronized,
):

    with condition:
        condition.wait()

    couples_memory = sm.SharedMemory(name="couples_mem")
    print(couples_memory)
    couples = np.ndarray(
        shape=(self.population_size // 4, 2),
        dtype=np.int64,
        buffer=couples_memory.buf,
    )

    print(f"{mp.current_process().name} ready")

    with condition:
        condition.wait_for(lambda: main_ready.value == 1)

    for c in couples:
        print(f"{mp.current_process().name}: {c}")

    with condition:
        process_ready.value += 1
        condition.notify_all()

    couples_memory.close()
