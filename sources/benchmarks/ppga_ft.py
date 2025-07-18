import os
import time

import numpy as np
import pandas as pd
from common import make_predictions, parse_args
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from neighborhood_generator import genetic
from ppga import algorithms, base, log

if __name__ == "__main__":
    # get CLI args
    args = parse_args()

    # set the log level
    logger = log.getUserLogger()
    logger.setLevel(args.log.upper())

    filepaths = os.listdir("datasets/")
    datasets = [
        pd.read_csv(f"datasets/{fp}") for fp in filepaths if fp.endswith("0.csv")
    ]
    classifiers = [RandomForestClassifier(), SVC(), MLPClassifier()]
    clf = classifiers[
        ["RandomForestClassifier", "SVC", "MLPClassifier"].index(args.model)
    ]

    ps = 4000
    # ps = 1000
    workers = [1, 2, 4, 8, 16, 32]
    # workers = [1, 2, 4]

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
        for d, df in enumerate(datasets):
            X, y = make_predictions(clf, df, 5)
            outcomes = np.unique(y)
            toolbox = genetic.create_toolbox(X)
            for i, (point, outcome) in enumerate(zip(X, y)):
                for target in outcomes:
                    logger.info(f"dataset {d + 1}/{len(datasets)}")
                    logger.info(f"point {i + 1}/{len(y)}")
                    logger.info(f"features: {len(point)}")
                    logger.info(f"class: {outcome}")
                    logger.info(f"target: {target}")
                    logger.info(f"classifier: {args.model}")
                    logger.info(f"population_size: {ps}")
                    logger.info(f"workers: {w}")

                    toolbox = genetic.update_toolbox(toolbox, point, int(target), clf)

                    times = []
                    ptimes = []
                    for _ in range(2):
                        hof = base.HallOfFame(ps)
                        start = time.process_time()
                        pop, stats = algorithms.simple(
                            toolbox, ps, 0.1, 0.7, 0.3, 10, hof, w
                        )
                        end = time.process_time()
                        ptime = np.sum(stats.times)
                        times.append((end - start) + ptime)
                        ptimes.append(ptime)

                    results["point"].append(i)
                    results["features"].append(len(point))
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
            f"results/performance/ppga_{args.model}_feature_{args.suffix}.csv",
            index=False,
            header=True,
        )
        print(results_df)
