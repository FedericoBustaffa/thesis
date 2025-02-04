import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    files = os.listdir("results/neighborhood/")
    files = [f for f in files if "MLPClassifier" in f]
    fps = [open(f"results/neighborhood/{f}") for f in files]

    chunks = []
    for fp in fps:
        chunk = json.load(fp)
        for c in chunk:
            chunks.append(c)
        fp.close()

    diversity = np.array([c["stats"]["diversity"] for c in chunks])
    plt.figure(figsize=(16, 9), dpi=200)
    plt.title("Biodiversity")
    plt.plot([i for i in range(100)], diversity.mean(axis=0))
    plt.grid()
    plt.show()

    mean_fit = np.array([c["stats"]["mean"] for c in chunks])
    plt.figure(figsize=(16, 9), dpi=200)
    plt.title("Mean fitness")
    plt.plot([i for i in range(100)], np.nanmean(mean_fit, axis=0))
    plt.grid()
    plt.show()
