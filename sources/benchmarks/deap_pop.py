import multiprocessing as mp
import time

import numpy as np
import pandas as pd
from common import make_predictions, parse_args
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from deap import algorithms, tools
from neighborhood_generator import genetic_deap as genetic
from ppga import log

if __name__ == "__main__":
    # get CLI args
    args = parse_args()

    # set the log level
    logger = log.getUserLogger()
    logger.setLevel(args.log.upper())

    df = pd.read_csv("datasets/classification_10010_32_2_1_0.csv")
    classifiers = [RandomForestClassifier(), SVC(), MLPClassifier()]
    clf = classifiers[
        ["RandomForestClassifier", "SVC", "MLPClassifier"].index(args.model)
    ]
    population_sizes = [1000, 2000, 4000, 8000, 16000]
    # population_sizes = [1000]
    workers = [1, 2, 4, 8, 16, 32]
    # workers = [4]

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

    X, y = make_predictions(clf, df, 10)
    outcomes = np.unique(y)
    toolbox = genetic.create_toolbox_deap(X)

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

                    toolbox = genetic.update_toolbox_deap(
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
                        start = time.process_time()
                        _, _, ptime = algorithms.eaSimple(
                            pop, toolbox, 0.8, 0.2, 20, None, hof
                        )
                        end = time.process_time()
                        times.append(end - start)
                        ptimes.append(ptime)

                        if pool is not None:
                            pool.close()
                            pool.join()

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
        f"results/deap_{args.model}_32_{args.suffix}.csv",
        index=False,
        header=True,
    )
    print(results_df)
