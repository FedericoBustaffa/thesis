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

    # set the logger
    logger = log.getUserLogger()
    logger.setLevel(args.log.upper())

    df = pd.read_csv("datasets/classification_10010_32_2_1_0.csv")
    classifiers = [RandomForestClassifier(), SVC(), MLPClassifier()]
    clf = classifiers[
        ["RandomForestClassifier", "SVC", "MLPClassifier"].index(args.model)
    ]
    # population_sizes = [1000, 2000, 4000, 8000, 16000]
    population_sizes = [1000, 2000]
    # workers = [1, 2, 4, 8, 16, 32]
    workers = [1, 4, 16]

    results = {
        "point": [],
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

    X, y = make_predictions(clf, df, 5)
    outcomes = np.unique(y)
    toolbox = genetic.create_toolbox(X)

    for w in workers:
        for ps in population_sizes:
            for i, (features, outcome) in enumerate(zip(X, y)):
                for target in outcomes:
                    logger.info(f"point {i + 1}/{len(y)}")
                    logger.info(f"features: {i}")
                    logger.info(f"class: {outcome}")
                    logger.info(f"target: {target}")
                    logger.info(f"classifier: {args.model}")
                    logger.info(f"population_size: {ps}")
                    logger.info(f"workers: {w}")

                    toolbox = genetic.update_toolbox(
                        toolbox, features, int(target), clf
                    )

                    times = []
                    ptimes = []  # only parallel time
                    for _ in range(10):
                        hof = base.HallOfFame(ps)
                        start = time.perf_counter()
                        pop, stats = algorithms.simple(
                            toolbox, ps, 0.1, 0.8, 0.2, 20, hof, w
                        )
                        end = time.perf_counter()
                        times.append(end - start)
                        ptimes.append(np.sum(stats.times))

                    results["point"].append(i)
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
        f"results/ppga_{args.model}_32_{args.suffix}.csv",
        index=False,
        header=True,
    )
    print(results_df)
