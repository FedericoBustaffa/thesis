import os

import blackbox
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from explain import explain

if __name__ == "__main__":
    blackboxes = [RandomForestClassifier(), SVC(), MLPClassifier()]

    filepaths = os.listdir("./datasets/")

    for filepath in filepaths:
        for bb in blackboxes:
            predictions = blackbox.make_predictions(bb, filepath)

            X = predictions[[k for k in predictions if k != "outcome"]].to_numpy()
            y = predictions["outcome"].to_numpy()

            stats = explain(X, y, bb, 1000)
            name = str(bb).removesuffix("()")

            params = filepath.split("_")
            samples = int(params[1])
            features = int(params[2])
            classes = int(params[3])
            clusters = int(params[4])
            seed = int(params[5].removesuffix(".csv"))

            stats.to_csv(
                f"datasets/{name}_{samples}_{features}_{classes}_{clusters}_{seed}.csv",
                header=True,
                index=False,
            )
