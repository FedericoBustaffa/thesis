import numpy as np
import pandas as pd
from common import draw_graph, evaluate

from ppga import algorithms, base, log, tools, utility

if __name__ == "__main__":
    log.setLevel("INFO")
    logger = log.getUserLogger()
    logger.setLevel("INFO")

    # Max generations
    G = 200

    ntowns = [25, 50, 100, 200, 400]
    population_sizes = [2000]
    df = {"towns": [], "distance": []}

    # sequential execution
    for tn in ntowns:
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

            hall_of_fame = base.HallOfFame(1)

            best, stats = algorithms.simple(
                toolbox=toolbox,
                population_size=ps,
                keep=0.2,
                cxpb=0.7,
                mutpb=0.3,
                max_generations=G,
                hall_of_fame=hall_of_fame,
                workers_num=4,
            )
            df["towns"].append(tn)
            df["distance"].append(evaluate(hall_of_fame[0].chromosome, towns)[0])
            logger.info(f"sequential best score: {best[0].fitness}")

    df = pd.DataFrame(df)
    df.to_csv("problems/tsp/results/ppga_tsp.csv", index=False, header=True)
    print(df)

    draw_graph(data, hall_of_fame[0].chromosome)
    utility.plot.fitness_trend(stats)
    utility.plot.biodiversity_trend(stats)
    utility.plot.evals(stats.evals)
