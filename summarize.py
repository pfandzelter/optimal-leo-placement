#
# Copyright (c) Tobias Pfandzelter. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import os
import concurrent.futures
import typing

import pandas as pd
import tqdm

import config

def get_mean_max(name: str):
    shell_names = {
        "st1": "Starlink A",
        "st2": "Starlink B",
        "ku1": "Kuiper A",
        "ku2": "Kuiper B",
    }


    d_df: typing.List[pd.DataFrame] = []

    for next_step in tqdm.trange(0, config.STEPS, config.INTERVAL, desc="Shell {}".format(shell_names[name])):
        results_file = os.path.join(config.PLACEMENT_DISTANCES_DIR, "{}-{}.csv".format(name, next_step))

        df = pd.read_csv(results_file)

        df["distance"] = df["distance"] / 1000.0
        df["pair"] = df["n"].astype(str) + "-" + df["rn"].astype(str)
        df["SLO"] = df["type"].astype(str) + "-" + df["d"].astype(str)
        df.drop(["type", "d", "n", "rn"], axis=1, inplace=True)

        d_df.append(df)

    df_s = pd.DataFrame(columns=["pair", "SLO"])
    df_s = pd.concat([df_s] + d_df)
    df_s["Shell"] = shell_names[name]
    print(df_s.head())

    df_mean = df_s.groupby(["pair", "SLO", "Shell"]).mean()
    df_mean.reset_index(inplace=True)
    df_mean.rename(columns={"distance": "mean"}, inplace=True)

    df_max = df_s.groupby(["pair", "SLO", "Shell"]).max()
    df_max.reset_index(inplace=True)
    df_max.rename(columns={"distance": "max"}, inplace=True)

    return (df_mean, df_max)


if __name__ == "__main__":
    results = pd.DataFrame(columns=["pair", "mean", "max", "Shell", "SLO"])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        r = executor.map(get_mean_max, [s["name"] for s in config.SHELLS])
        for df_mean, df_max in r:
            results = results.append(df_mean)
            results = results.append(df_max)
            results.reset_index(inplace=True, drop=True)

    results.to_csv(config.RESULTS_FILE, index=False)
