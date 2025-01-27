import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
from common import make_predictions, parse_args
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from deap import algorithms, tools
from neighborhood_generator import genetic_deap
from ppga import log

if __name__ == "__main__":
    # get CLI args
    args = parse_args()

    # set the log level
    logger = log.getUserLogger()
    logger.setLevel(args.log.upper())

    filepaths = os.listdir("datasets/")
    datasets = [pd.read_csv(f"datasets/{fp}") for fp in filepaths]
    classifiers = [RandomForestClassifier(), SVC(), MLPClassifier()]
    clf = classifiers[
        ["RandomForestClassifier", "SVC", "MLPClassifier"].index(args.model)
    ]

    ps = 1000
    workers = [1, 2, 4, 8, 16, 32]
    # workers = [4]

    results = {
        "point": [],
        "features": [],
        "class": [],
        "target": [],
        "classifier": [],
        "population_size": [],
        "workers": [],
        "time": [],
        "time_std": [],
        "ptime": [],
        "ptime_std": [],
    }

    for w in workers:
        for df in datasets:
            X, y = make_predictions(clf, df, 10)
            outcomes = np.unique(y)
            toolbox = genetic_deap.create_toolbox_deap(X)
            for i, (features, outcome) in enumerate(zip(X, y)):
                for target in outcomes:
                    logger.info(f"point {i + 1}/{len(y)}")
                    logger.info(f"features: {len(X[0])}")
                    logger.info(f"class: {outcome}")
                    logger.info(f"target: {target}")
                    logger.info(f"classifier: {args.model}")
                    logger.info(f"population_size: {ps}")
                    logger.info(f"workers: {w}")

                    toolbox = genetic_deap.update_toolbox_deap(
                        toolbox, features, int(target), clf
                    )

                    times = []
                    ptimes = []
                    for i in range(10):
                        pool = None
                        if w > 1:
                            pool = mp.Pool(w)
                            toolbox.register("map", pool.map)
                            # toolbox.register("map", pool.map, chunksize=ps // w)
                        else:
                            toolbox.register("map", map)

                        pop = toolbox.population(n=ps)
                        hof = tools.HallOfFame(int(0.1 * ps), similar=np.array_equal)
                        start = time.perf_counter()
                        _, _, ptime = algorithms.eaSimple(
                            pop, toolbox, 0.8, 0.2, 20, None, hof
                        )
                        end = time.perf_counter()
                        times.append(end - start)
                        ptimes.append(ptime)

                        if pool is not None:
                            pool.close()
                            pool.join()

                    results["point"].append(i)
                    results["features"].append(len(X[0]))
                    results["class"].append(outcome)
                    results["target"].append(target)

                    results["classifier"].append(str(clf).removesuffix("()"))
                    results["population_size"].append(ps)
                    results["workers"].append(w)

                    # total work time
                    results["time"].append(np.mean(times))
                    results["time_std"].append(np.std(times))

                    # only parallel time
                    results["ptime"].append(np.mean(ptimes))
                    results["ptime_std"].append(np.std(ptimes))

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        f"results/deap_{args.model}_32_{args.suffix}.csv",
        index=False,
        header=True,
    )
    print(results_df)
