import multiprocessing as mp


class Synchronizer:
    def __init__(self, workers_num):
        self.main_ready = mp.Condition()
        self.worker_ready = mp.Condition()
        self.worker_ready_count = mp.Value("i", workers_num)

    def wait_for_main(self):
        with self.main_ready:
            self.main_ready.wait()

    def wait_for_workers(self):
        with self.worker_ready_count:
            while self.worker_ready_count.value > 0:
                with self.worker_ready:
                    self.worker_ready.wait()

    def worker_done(self):
        with self.worker_ready_count:
            self.worker_ready.acquire()
