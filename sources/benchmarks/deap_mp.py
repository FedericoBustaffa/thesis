import argparse
import multiprocessing as mp
import random
import time

import numpy as np
import pandas as pd
from deap import base, creator, tools
from parallelism import make_predictions
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from neighborhood_generator import genetic
from ppga import log


def create_toolbox(X: np.ndarray) -> base.Toolbox:
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

    return toolbox


def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(
                offspring[i - 1], offspring[i]
            )
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            (offspring[i],) = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def eaSimple(population, toolbox, cxpb, mutpb, ngen, halloffame):
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    ptime = 0.0
    # Begin the generational process
    for gen in range(1, ngen + 1):
        print(f"generation: {gen}")
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        start = time.perf_counter()
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        ptime += time.perf_counter() - start
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

    return ptime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        help="specify the model to benchmark",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        help="specify the logging level",
    )

    args = parser.parse_args()

    # set the log level
    logger = log.getUserLogger()
    logger.setLevel(args.log.upper())

    df = pd.read_csv("datasets/classification_100_32_2_1_0.csv")
    classifiers = [RandomForestClassifier(), SVC(), MLPClassifier()]
    clf = classifiers[
        ["RandomForestClassifier", "SVC", "MLPClassifier"].index(args.model)
    ]
    population_sizes = [1000, 2000, 4000, 8000, 16000]
    workers = [1, 2, 4, 8, 16, 32]

    results = {
        "classifier": [],
        "population_size": [],
        "workers": [],
        "time": [],
        "time_std": [],
        "ptime": [],
        "ptime_std": [],
    }

    X, y = make_predictions(clf, df, 0.3)

    toolbox = create_toolbox(X)

    for w in workers:
        pool = None
        if w > 1:
            pool = mp.Pool(w)
            toolbox.register("map", pool.map)
        else:
            toolbox.register("map", map)

        for ps in population_sizes:
            logger.info(f"classifier: {args.model}")
            logger.info(f"population size: {ps}")
            logger.info(f"workers: {w}")
            times = []
            ptimes = []
            for i in range(10):
                pop = getattr(toolbox, "population")(n=ps)
                hof = tools.HallOfFame(ps, similar=np.array_equal)
                start = time.perf_counter()
                ptime = eaSimple(pop, toolbox, 0.8, 0.2, 5, hof)
                end = time.perf_counter()
                times.append(end - start)
                ptimes.append(ptime)

            results["classifier"].append(str(clf).removesuffix("()"))
            results["population_size"].append(ps)
            results["workers"].append(w)

            # total work time
            results["time"].append(np.mean(times))
            results["time_std"].append(np.std(times))

            # only parallel time
            results["ptime"].append(np.mean(ptimes))
            results["ptime_std"].append(np.std(ptimes))

        if pool is not None:
            pool.join()
            pool.close()

    results = pd.DataFrame(results)
    results.to_csv(
        f"results/deap_benchmark_{args.model}_32.csv",
        index=False,
        header=True,
    )
    print(results)
