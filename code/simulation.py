import sys

from colorama import Fore

import pipe
import sequential
import shared

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

        print(Fore.CYAN + f"sequential")
        sequential.main(sys.argv[1:])

        for w in range(workers):
            sys.argv[-1] = w + 1
            print(Fore.YELLOW + f"pipe with {w + 1} workers")
            pipe.main(sys.argv[1:])

        for w in range(workers):
            sys.argv[-1] = w + 1
            print(Fore.MAGENTA + f"shared memory with {w + 1} workers")
            shared.main(sys.argv[1:])

        print(Fore.GREEN + f"simulation {i+1} ended")
