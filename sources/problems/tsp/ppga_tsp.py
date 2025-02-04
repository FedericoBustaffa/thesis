import numpy as np
import pandas as pd
from common import evaluate

from ppga import algorithms, base, log, tools, utility

if __name__ == "__main__":
    log.setLevel("INFO")
    logger = log.getUserLogger()
    logger.setLevel("INFO")

    # Max generations
    G = 500

    towns_num = [10, 20, 50, 100]
    towns_num = [20]
    population_sizes = [100, 200, 400, 800]
    population_sizes = [500]
    df = {"towns": [], "population_size": [], "distance": []}

    # sequential execution
    for tn in towns_num:
        for ps in population_sizes:
            data = pd.read_csv(f"problems/tsp/datasets/towns_{tn}.csv")
            x_coords = data["x"]
            y_coords = data["y"]
            towns = np.array([[x, y] for x, y in zip(x_coords, y_coords)])

            toolbox = base.ToolBox()
            toolbox.set_weights((-1.0,))
            toolbox.set_generation(tools.gen_permutation, range(len(towns)))
            toolbox.set_selection(tools.sel_tournament, tournsize=2)
            toolbox.set_crossover(tools.cx_one_point_ordered)
            toolbox.set_mutation(tools.mut_rotation)
            toolbox.set_evaluation(evaluate, towns)

            hall_of_fame = base.HallOfFame(5)

            best, stats = algorithms.simple(
                toolbox=toolbox,
                population_size=ps,
                keep=0.2,
                cxpb=0.7,
                mutpb=0.3,
                max_generations=G,
                hall_of_fame=hall_of_fame,
            )
            df["towns"].append(tn)
            df["population_size"].append(ps)
            df["distance"].append(evaluate(hall_of_fame[0].chromosome, towns)[0])
            logger.info(f"sequential best score: {best[0].fitness}")

    df = pd.DataFrame(df)
    # df.to_csv("problems/tsp/results/ppga_tsp.csv", index=False, header=True)
    print(df)

    # utility.plot.draw_graph(data, hall_of_fame[0].chromosome)
    utility.plot.fitness_trend(stats)
    utility.plot.biodiversity_trend(stats)
    utility.plot.evals(stats.evals)
