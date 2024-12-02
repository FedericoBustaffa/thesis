import json
import os

import pandas as pd


def read_file(filepath: str) -> list[dict]:
    file = open(filepath, "r")
    lines = file.readlines()
    file.close()

    data = []
    for line in lines:
        if "BENCHMARK" in line:
            data.append(json.loads(line))

    return data


def parse_values(lines: list[dict]) -> pd.DataFrame:
    stats = {"process_name": [], "field": [], "time": []}
    for line in lines:
        stats["process_name"].append(line["process_name"])
        stats["field"].append(line["field"])
        stats["time"].append(float(line["time"]))

    return pd.DataFrame(stats)


def main():
    # backups results dir
    if "results" not in os.listdir("."):
        os.mkdir("results")

    # sequential simulation
    lines = read_file("logs/sequential.json")
    stats = parse_values(lines)
    stats.to_csv("results/sequential.csv", header=True, index=False)

    # parallel simulation
    lines = read_file("logs/parallel.json")
    pstats = parse_values(lines)
    pstats.to_csv("results/parallel.csv", header=True, index=False)

    stime = stats[stats["field"] == "stime"]["time"].sum()
    ptime = pstats[pstats["field"] == "ptime"]["time"].sum()

    # total time
    print(f"sequential time: {stime} seconds")
    print(f"parallel time: {ptime} seconds")
    print(f"speed up: {stime / ptime}")

    # take only the parallelized part
    cx_mut_eval_time = stats[stats["field"] == "cx_mut_eval"]["time"].sum()
    print(f"sequential cx + mut + eval time: {cx_mut_eval_time} seconds")

    # sync + work time
    parallel_time = pstats[pstats["field"] == "parallel"]["time"].sum()
    print(f"parallel cx + mut + eval time: {parallel_time} seconds")

    # approximate sync time
    # print(f"sync time: {parallel_time - worst_worker} seconds")

    # pure speed up
    print(f"pure speed up: {cx_mut_eval_time / parallel_time}")


if __name__ == "__main__":
    main()
