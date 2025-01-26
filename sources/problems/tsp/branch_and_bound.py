import argparse
import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from utils import plotting

from ppga import log


def held_karp_tsp(graph):
    # Numero di nodi
    n = len(graph.nodes)

    # Lista dei nodi
    nodes = list(graph.nodes)

    # Matrice delle distanze (da usare nella programmazione dinamica)
    distance_matrix = nx.to_numpy_array(graph)

    # Distanza minima per ogni sottoinsieme di nodi e nodo finale
    dp = {}
    for i in range(1, n):  # Inizializzazione: si parte dal nodo 0
        dp[(1 << i, i)] = distance_matrix[0][i]

    # Programmazione dinamica per sottoinsiemi di dimensione crescente
    for subset_size in range(2, n):  # Sottinsiemi di dimensione 2 fino a n-1
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = sum(1 << i for i in subset)
            for j in subset:
                dp[(bits, j)] = min(
                    dp[(bits ^ (1 << j), k)] + distance_matrix[k][j]
                    for k in subset
                    if k != j
                )

    # Aggiungere il nodo di partenza (0) per completare il ciclo
    bits = (1 << n) - 1  # Tutti i nodi visitati
    result = min(dp[(bits ^ (1 << i), i)] + distance_matrix[i][0] for i in range(1, n))

    # Ricostruzione del percorso
    path = [0]  # Partiamo dal nodo iniziale
    last = 0
    while len(path) < n:
        next_node = min(
            (
                (dp[(bits, k)] + distance_matrix[k][last], k)
                for k in range(1, n)
                if bits & (1 << k)
            ),
            key=lambda x: x[0],
        )[1]
        path.append(next_node)
        bits ^= 1 << next_node
        last = next_node

    return result, path


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
    towns = np.array([np.array([x, y]) for x, y in zip(x_coords, y_coords)])

    # Creazione del grafo
    G = nx.complete_graph(args.towns_num)
    pos = nx.spring_layout(G)
    for u, v in G.edges():
        G.edges[u, v]["weight"] = np.linalg.norm(
            np.array(towns[u]) - np.array(towns[v])
        )

    # Risoluzione tramite Christofides
    results, path = held_karp_tsp(G)

    print("Lunghezza approssimata del percorso:", results)
    print("Percorso:", path)

    plotting.draw_graph(data, path)
