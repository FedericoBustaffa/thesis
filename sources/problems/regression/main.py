import deap_regression
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ppga_regression

if __name__ == "__main__":
    results = {
        "samples": [],
        "numpy_MSE": [],
        "deap_MSE": [],
        "ppga_MSE": [],
    }

    matplotlib.rcParams.update({"font.size": 16})

    n_samples = [50, 100, 200, 400, 800]

    for n in n_samples:
        # random samples generation
        x = np.linspace(-100, 100, n)
        y = np.random.normal(-10, 10, (n,)) * x + np.random.normal(-100, 100, (n,))
        points = np.stack((x, y), axis=1)

        # Numpy regression
        m, q = np.polyfit(x, y, 1)
        numpy_y = m * x + q
        numpy_mse = np.mean((y - numpy_y) ** 2)

        # DEAP regression
        m, q = deap_regression.linear_regression(points)
        deap_y = m * x + q
        deap_mse = np.mean((y - deap_y) ** 2)

        # PPGA regression
        m, q = ppga_regression.linear_regression(points)
        ppga_y = m * x + q
        ppga_mse = np.mean((y - ppga_y) ** 2)

        results["samples"].append(n)
        results["numpy_MSE"].append(numpy_mse)
        results["deap_MSE"].append(deap_mse)
        results["ppga_MSE"].append(ppga_mse)

        plt.figure(figsize=(8, 4), dpi=300)
        # plt.title("Regressione lineare genetica")
        plt.scatter(x, y, ec="w")
        plt.plot(x, numpy_y, c="g", label="Numpy")
        plt.plot(x, deap_y, c="r", label="DEAP")
        plt.plot(x, ppga_y, c="b", label="PPGA")

        plt.grid()
        plt.legend()
        plt.tight_layout()
        # plt.savefig("/home/federico/tesi/immagini/regression.svg")
        plt.show()

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(
        "problems/regression/results/genetic_regression.csv", index=False, header=True
    )
