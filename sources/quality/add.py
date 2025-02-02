import argparse

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lib",
        type=str,
        choices={"ppga", "deap"},
        help="specify the lib dataset",
    )
    args = parser.parse_args()

    lib_df = pd.read_csv(f"results/quality/{args.lib}2.csv")
    lib_df = lib_df[lib_df["model"] == "RandomForestClassifier"]
    lib_df = lib_df[lib_df["population_size"] == 1000]
    print(lib_df)
    # lib_df.to_csv(f"results/quality/{args.lib}2.csv", header=True, index=False)
