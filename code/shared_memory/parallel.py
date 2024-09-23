import multiprocessing as mp
import multiprocessing.shared_memory as sm
import multiprocessing.sharedctypes as st
import multiprocessing.synchronize as sync
import time

import numpy as np


def share(buffer, mem_name):

    buffer = np.array(buffer)
    buffer_memory = sm.SharedMemory(name=mem_name, create=True, size=buffer.nbytes)

    shared_buffer = np.ndarray(
        shape=buffer.shape,
        dtype=buffer.dtype,
        buffer=buffer_memory.buf,
    )
    np.copyto(shared_buffer, buffer)

    return buffer_memory, shared_buffer


def start_workers(self):
    self.workers = [
        mp.Process(
            target=self.parallel_work,
            args=[
                i,
                self.workers_num,
                self.shapes,
                self.dtypes,
                self.main_ready[i],
                self.workers_ready[i],
                self.stops[i],
            ],
        )
        for i in range(self.workers_num)
    ]

    for w in self.workers:
        w.start()


def parallel_work(
    self,
    index: int,
    workers_num: int,
    shapes,
    dtypes,
    main_ready: sync.Event,
    ready: sync.Event,
    stop: st.Synchronized,
):
    couples_memory = sm.SharedMemory(name="couples_mem")
    couples = np.ndarray(
        shape=shapes[0],
        dtype=dtypes[0],
        buffer=couples_memory.buf,
    )

    population_memory = sm.SharedMemory(name="population_mem")
    population = np.ndarray(
        shape=shapes[1],
        dtype=dtypes[1],
        buffer=population_memory.buf,
    )

    scores_memory = sm.SharedMemory(name="scores_mem")
    scores = np.ndarray(
        shape=shapes[2],
        dtype=dtypes[2],
        buffer=scores_memory.buf,
    )

    offsprings_memory = sm.SharedMemory(name="offsprings_mem")
    offsprings = np.ndarray(
        shape=shapes[3],
        dtype=dtypes[3],
        buffer=offsprings_memory.buf,
    )

    offsprings_scores_memory = sm.SharedMemory(name="offsprings_scores_mem")
    offsprings_scores = np.ndarray(
        shape=shapes[4],
        dtype=dtypes[4],
        buffer=offsprings_scores_memory.buf,
    )

    chunk_size = len(couples) // workers_num
    # print(f"{mp.current_process().name} chunk size: {chunk_size}")
    while True:

        main_ready.wait()
        main_ready.clear()

        with stop:
            if stop.value == 1:
                break

        for i in range(index * chunk_size, index * chunk_size + chunk_size, 1):
            start = time.perf_counter()
            father = population[couples[i][0]].view()
            mother = population[couples[i][1]].view()
            nano_start = time.perf_counter_ns()
            offspring1, offspring2 = self.crossover_func(father, mother)
            nano_end = time.perf_counter_ns()
            self.timings["crossover"] += time.perf_counter() - start
            if self.timings["crossover_operator"] < (nano_end - nano_start):
                self.timings["crossover_operator"] = nano_end - nano_start
                
            start = time.perf_counter()
            if np.random.random() < self.mutation_rate:
                offspring1 = np.array(self.mutation_func(offspring1))

            if np.random.random() < self.mutation_rate:
                offspring2 = np.array(self.mutation_func(offspring2))
                
            np.copyto(offsprings[i * 2], offspring1)
            np.copyto(offsprings[i * 2 + 1], offspring2)
            self.timings["mutation"] += time.perf_counter() - start
                
            start = time.perf_counter()
            offsprings_scores[i * 2] = self.fitness_func(offspring1)
            offsprings_scores[i * 2 + 1] = self.fitness_func(offspring2)
            self.timings["evaluation"] += time.perf_counter() - start

        ready.set()
    
    
    print(f"{mp.current_process().name} crossover time: {self.timings["crossover"]}")
    print(f"{mp.current_process().name} crossover operator max time: {self.timings["crossover_operator"]} ns")
    print(f"{mp.current_process().name} mutation time: {self.timings["mutation"]}")
    print(f"{mp.current_process().name} evaluation time: {self.timings["evaluation"]}")

    couples_memory.close()
