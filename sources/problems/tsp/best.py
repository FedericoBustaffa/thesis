import argparse

import networkx as nx
import numpy as np
import pandas as pd
from common import draw_graph, evaluate
from networkx.algorithms.approximation import traveling_salesman_problem

from ppga import log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="INFO", help="specify the log level")
    args = parser.parse_args()

    # set logging
    log.setLevel(args.log.upper())
    logger = log.getUserLogger()
    logger.setLevel("INFO")

    ntowns = [25, 50, 100, 200, 400]
    results = {"towns": [], "distance": []}

    for n in ntowns:
        data = pd.read_csv(f"problems/tsp/datasets/towns_{n}.csv")
        x_coords = data["x"]
        y_coords = data["y"]
        towns = np.array([np.array([x, y]) for x, y in zip(x_coords, y_coords)])

        # Creazione del grafo
        G = nx.complete_graph(n)
        pos = nx.spring_layout(G)
        for u, v in G.edges():
            G.edges[u, v]["weight"] = np.linalg.norm(
                np.array(towns[u]) - np.array(towns[v])
            )

        # Risoluzione tramite Christofides
        path = traveling_salesman_problem(G, cycle=False)

        d = evaluate(path, towns)[0]
        print(f"Lunghezza approssimata del percorso: {d:.2f}")
        # print(f"Percorso: {path}")

        results["towns"].append(n)
        results["distance"].append(d)

        if args.log == "debug":
            draw_graph(data, path)

    df = pd.DataFrame(results)
    # df.to_csv("problems/tsp/results/nx_tsp.csv", index=False, header=True)
    print(df)
