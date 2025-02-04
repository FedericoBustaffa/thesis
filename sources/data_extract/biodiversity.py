import json
import os

import pandas as pd

if __name__ == "__main__":
    files = os.listdir("results/neighborhood/")
    files = [f for f in files if "MLPClassifier" in f]
    fps = [open(f"results/neighborhood/{f}") for f in files]

    for fp in fps:
        fp.close()
