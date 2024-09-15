import multiprocessing as mp
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory


def task(pipe: Connection):
    buffer = pipe.recv()
    for i in range(len(buffer)):
        buffer[i] += 1
    pipe.send(buffer)

    return


def task2(mem_name: str):
    mem = SharedMemory(name=mem_name, create=False)
    print(f"{mem.buf[:]}")
    mem.close()


if __name__ == "__main__":
    buffer1 = [i for i in range(10)]
    buffer2 = [i for i in range(20)]

    # pipe11, pipe12 = mp.Pipe()
    # pipe21, pipe22 = mp.Pipe()

    mem1 = SharedMemory(name="mem1", create=True, size=2048)
    mem2 = SharedMemory(name="mem2", create=True, size=4096)

    p1 = mp.Process(target=task2, args=["mem1"])
    p2 = mp.Process(target=task2, args=["mem2"])

    # pipe11.send(buffer1)
    # pipe21.send(buffer2)

    p1.start()
    p2.start()

    # buffer1 = pipe11.recv()
    # buffer2 = pipe21.recv()

    p1.join()
    p2.join()

    mem1.unlink()
    mem2.unlink()

    print("buffer 1")
    for i in buffer1:
        print(i)

    print("buffer 2")
    for i in buffer2:
        print(i)
