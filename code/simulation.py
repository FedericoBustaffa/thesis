import sys
import time

import pipe
import sequential
import shared
from colorama import Fore

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(
            Fore.RED
            + f"USAGE: py {sys.argv[0]} <simulations> <cities> <population_size> <max_generations> <mutation_rate> <workers>"
        )
        exit(1)

    simulations = int(sys.argv[1])
    workers = int(sys.argv[-1])

    for i in range(simulations):
        print(Fore.GREEN + f"starting simulation: {i+1}")

        start = time.perf_counter()
        sequential.main(sys.argv[1:])
        print(Fore.GREEN + f"sequential: {time.perf_counter() - start}")

        for w in range(2, workers + 1, 1):
            sys.argv[-1] = str(w)
            start = time.perf_counter()
            pipe.main(sys.argv[1:])
            print(Fore.GREEN + f"pipe with {w} workers: {time.perf_counter() - start}")

        for w in range(2, workers + 1, 1):
            sys.argv[-1] = str(w)
            start = time.perf_counter()
            shared.main(sys.argv[1:])
            print(
                Fore.GREEN
                + f"shared memory with {w} workers: {time.perf_counter() - start}"
            )

        print(Fore.GREEN + f"simulation {i+1} ended")
