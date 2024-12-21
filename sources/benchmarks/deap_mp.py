import multiprocessing as mp
import time

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from explain import genetic
from ppga import log


def make_predictions(model, data: pd.DataFrame, test_size: float = 0.3):
    features_index = [col for col in data.columns if col.startswith("feature_")]
    X = data[features_index].to_numpy()
    y = data["outcome"].to_numpy()

    # split train and test set
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    # train the model
    model.fit(X_train, y_train)

    # these will be the data to explain
    to_explain = np.asarray(model.predict(X_test))

    return np.asarray(X_test), to_explain


if __name__ == "__main__":
    logger = log.getUserLogger()
    logger.setLevel("INFO")

    df = pd.read_csv("datasets/classification_100_32_2_1_0.csv")
    classifiers = [RandomForestClassifier(), SVC(), MLPClassifier()]
    population_sizes = [1000, 2000, 4000, 8000, 16000]
    workers = [1, 2, 4, 8, 16, 32]

    results = {
        "classifier": [],
        "population_size": [],
        "workers": [],
        "time": [],
        "time_std": [],
    }

    for clf in classifiers:
        X, y = make_predictions(clf, df, 0.3)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=getattr(creator, "FitnessMin"))
        toolbox = base.Toolbox()
        point = X[0]
        target = y[0]
        toolbox.register("features", np.copy, point)
        toolbox.register(
            "individual",
            tools.initIterate,
            getattr(creator, "Individual"),
            getattr(toolbox, "features"),
        )

        toolbox.register(
            "population", tools.initRepeat, list, getattr(toolbox, "individual")
        )

        toolbox.register(
            "evaluate", genetic.evaluate, point=point, target=target, blackbox=clf
        )
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=X.mean(),
            sigma=X.std(),
            indpb=0.5,
        )
        for ps in population_sizes:
            for w in workers:
                times = []
                if w == 1:
                    toolbox.register("map", map)
                else:
                    toolbox.register("map", mp.Pool(w).map)

                for i in range(10):
                    pop = getattr(toolbox, "population")(n=ps)
                    hof = tools.HallOfFame(ps, similar=np.array_equal)
                    start = time.perf_counter()
                    algorithms.eaSimple(pop, toolbox, 0.8, 0.2, 5, hof)
                    end = time.perf_counter()
                    times.append(end - start)

                results["classifier"].append(str(clf).removesuffix("()"))
                results["population_size"].append(ps)
                results["workers"].append(w)
                results["time"].append(np.mean(times))
                results["time_std"].append(np.std(times))
                logger.info(f"classifier: {str(clf).removesuffix('()')}")
                logger.info(f"population_size: {ps}")
                logger.info(f"workers: {w}")

    results = pd.DataFrame(results)
    results.to_csv("datasets/deap_mp_32.csv", index=False, header=True)
    print(results)
