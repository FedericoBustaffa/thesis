import os

import blackbox
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from explain import explain
from ppga import log

if __name__ == "__main__":
    blackboxes = [RandomForestClassifier(), SVC(), MLPClassifier()]

    filepaths = sorted(os.listdir("./datasets/"))
    logger = log.getUserLogger()
    logger.setLevel("INFO")

    population_sizes = [1000, 4000, 16000]

    for filepath in filepaths:
        for bb in blackboxes:
            for ps in population_sizes:
                predictions = blackbox.make_predictions(bb, f"datasets/{filepath}")

                name = str(bb).removesuffix("()")
                logger.info(
                    f"{filepath} explained with {name} and {ps} synthetic individuals"
                )

                X = predictions[[k for k in predictions if k != "outcome"]].to_numpy()
                y = predictions["outcome"].to_numpy()

                stats = explain(X, y, bb, ps)

                params = filepath.split("_")
                samples = int(params[1])
                features = int(params[2])
                classes = int(params[3])
                clusters = int(params[4])
                seed = int(params[5].removesuffix(".csv"))

                stats.to_csv(
                    f"datasets/{name}_{ps}_{samples}_{features}_{classes}_{clusters}_{seed}.csv",
                    header=True,
                    index=False,
                )
