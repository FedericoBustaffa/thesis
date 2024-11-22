import random
import sys

import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"USAGE: py {sys.argv[0]} <T>")
        exit(1)

    T = int(sys.argv[1])

    # generate towns positions
    print(f"generating {T} towns...")
    data = pd.DataFrame(
        {
            "x": [random.random() for _ in range(T)],
            "y": [random.random() for _ in range(T)],
        }
    )
    print(f"{T} towns generated")
    data.to_csv(path_or_buf=f"datasets/towns_{T}.csv", sep=",", index=False)
    print(f"File generated: datasets/towns_{T}.csv")
