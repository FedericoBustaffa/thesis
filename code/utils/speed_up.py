import pandas as pd

if __name__ == "__main__":
    timings = pd.read_csv("shared_memory_timings.csv")
    print(timings)
