import json

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    same = {"point": [], "class": [], "target": [], "neighborhood": []}
    diff = {"point": [], "class": [], "target": [], "neighborhood": []}
    with open("problems/data_generation/synthetic.json", "r") as fp:
        data = json.load(fp)

        for res in data:
            if res["class"] == res["target"]:
                same["point"].append(res["point"])
                same["class"].append(res["class"])
                same["target"].append(res["target"])
                same["neighborhood"].append(res["neighborhood"])
            else:
                diff["point"].append(res["point"])
                diff["class"].append(res["class"])
                diff["target"].append(res["target"])
                diff["neighborhood"].append(res["neighborhood"])

    same["point"] = np.array(same["point"])
    same["class"] = np.array(same["class"])
    same["target"] = np.array(same["target"])

    diff["point"] = np.array(diff["point"])
    diff["diff"] = np.array(diff["class"])
    diff["target"] = np.array(diff["target"])

    mask = same["class"] == 0
    reds = same["point"][mask]
    blues = same["point"][~mask]

    plt.figure(figsize=(16, 9), dpi=300)
    plt.scatter(reds.T[0], reds.T[1], c="r", ec="w")
    plt.scatter(blues.T[0], blues.T[1], c="b", ec="w")

    ref = 0
    plt.scatter(blues.T[0][ref], blues.T[1][ref], c="g", ec="w", marker="X")

    # same_neigbors = np.array([nh["chromosome"] for nh in same["neighborhood"][0]])
    # plt.scatter(same_neigbors.T[0], same_neigbors.T[1], c="y", ec="w")

    diff_neighbors = np.array([nh["chromosome"] for nh in diff["neighborhood"][0]])
    plt.scatter(diff_neighbors.T[0], diff_neighbors.T[1], c="y", ec="w")

    plt.show()
