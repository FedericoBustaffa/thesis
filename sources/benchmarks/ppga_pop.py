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

    df = pd.read_csv("datasets/classification_10010_64_2_1_0.csv")
    classifiers = [RandomForestClassifier(), SVC(), MLPClassifier()]
    clf = classifiers[
        ["RandomForestClassifier", "SVC", "MLPClassifier"].index(args.model)
    ]
    population_sizes = [1000, 2000, 4000, 8000, 16000]
    workers = [1, 2, 4, 8, 16, 32]

    results = {
        "point": [],
        "features": [],
        "class": [],
        "target": [],
        "classifier": [],
        "population_size": [],
        "workers": [],
        "time": [],
        "ptime": [],
    }

    X, y = make_predictions(clf, df, 2)
    outcomes = np.unique(y)
    toolbox = genetic.create_toolbox(X)

    for w in workers:
        for ps in population_sizes:
            for i, (point, outcome) in enumerate(zip(X, y)):
                for target in outcomes:
                    logger.info(f"point {i + 1}/{len(y)}")
                    logger.info(f"features: {len(point)}")
                    logger.info(f"class: {outcome}")
                    logger.info(f"target: {target}")
                    logger.info(f"classifier: {args.model}")
                    logger.info(f"population_size: {ps}")
                    logger.info(f"workers: {w}")

                    toolbox = genetic.update_toolbox(toolbox, point, int(target), clf)

                    hof = base.HallOfFame(ps)
                    start = time.perf_counter()
                    pop, stats = algorithms.simple(
                        toolbox, ps, 0.1, 0.7, 0.3, 15, hof, w
                    )
                    end = time.perf_counter()
                    ptime = np.sum(stats.times)

                    results["point"].append(i)
                    results["features"].append(len(point))
                    results["class"].append(outcome)
                    results["target"].append(target)

                    results["classifier"].append(str(clf).removesuffix("()"))
                    results["population_size"].append(ps)
                    results["workers"].append(w)

                    results["time"].append(end - start)
                    results["ptime"].append(ptime)

            df = pd.DataFrame(results)
            df.to_csv(
                f"results/performance/ppga_{args.model}_pop_{args.suffix}.csv",
                index=False,
                header=True,
            )
            print(df)
