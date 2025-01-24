import argparse
from itertools import permutations

import numpy as np
import pandas as pd
from common import Town, evaluate
from utils import plotting

from ppga import log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("towns_num", type=int, help="choose the number of towns")
    args = parser.parse_args()

    # set logging
    log.setLevel("INFO")
    logger = log.getUserLogger()
    logger.setLevel("INFO")

    data = pd.read_csv(f"problems/tsp/datasets/towns_{args.towns_num}.csv")
    x_coords = data["x"]
    y_coords = data["y"]
    towns = [Town(x, y) for x, y in zip(x_coords, y_coords)]

    paths = np.array(
        [list(p) for p in permutations([i for i in range(args.towns_num)])]
    )

    distances = np.array([evaluate(p, towns)[0] for p in paths])
    indices = np.argsort(distances)

    paths = paths[indices]
    distances = paths[indices]

    plotting.draw_graph(data, paths[0])
