import argparse

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lib",
        type=str,
        choices={"ppga", "deap"},
        help="specify the library results to unify",
    )
    args = parser.parse_args()

    rf = pd.read_csv(f"results/quality/{args.lib}_RandomForestClassifier.csv")
    svm = pd.read_csv(f"results/quality/{args.lib}_SVC.csv")
    mlp = pd.read_csv(f"results/quality/{args.lib}_MLPClassifier.csv")

    df = pd.concat([rf, svm, mlp], axis=0)
    df.sort_values(
        by=["model", "features", "seed", "population_size", "point", "class", "target"],
        inplace=True,
    )

    df[["min_fitness", "mean_fitness", "max_fitness"]] *= -1.0
    df = df.rename(
        columns={
            "min_fitness": "max_distance",
            "mean_fitness": "mean_distance",
            "fitness_std": "distance_std",
            "max_fitness": "min_distance",
        }
    ).reset_index(drop=True)

    print(df)
    df.to_csv(f"results/quality/{args.lib}.csv", header=True, index=False)
