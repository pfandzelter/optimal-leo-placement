#
# Copyright (c) Tobias Pfandzelter. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import os

import matplotlib
import pandas as pd
import seaborn as sns

import config

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#matplotlib.rcParams['text.usetex'] = True
sns.set(rc={'figure.figsize':(6,5)}, style="whitegrid", font="CMU Sans Serif")

def save_fig(ax, name, file_type="pdf"):
    fig = ax.get_figure()
    fig.tight_layout()
    file_name = name + "." + file_type
    fig.savefig(os.path.join(config.GRAPHS_DIR, file_name), bbox_inches='tight')
    fig.clear()

if __name__ == "__main__":

    results = pd.read_csv(config.RESULTS_FILE)

    slo_names = {
        "hops-1.0": "1-Hop",
        "hops-4.0": "4-Hops",
        "mean-2997.92458": "Mean\n10ms",
        "max-2997.92458": "Max.\n10ms",
        "mean-29979.2458": "Mean\n100ms",
        "max-29979.2458": "Max.\n100ms",
    }

    results["SLO"] = results["SLO"].apply(lambda x: slo_names[x])

    sort_key = [
        "1-Hop",
        "4-Hops",
        "Mean\n10ms",
        "Max.\n10ms",
        "Mean\n100ms",
        "Max.\n100ms",
    ]

    results["SLO"] = pd.Categorical(results["SLO"], sort_key)

    results.sort_values("SLO")

    ax_mean = sns.boxplot(x="SLO", y="mean", hue="Shell", data=results, fliersize=1, zorder=20, boxprops={'zorder': 20})
    ax_mean.set_xlabel("SLO")
    ax_mean.set_ylabel("Mean Distance to Resource Node [km]")
    ax_mean.legend(loc="upper center", ncol = 4, borderaxespad=-2)
    ax_mean.axhline(y=2997.92458, c="black", linestyle="--", zorder=10)
    ax_mean.text(5.6, 2600, "10ms", bbox=dict(facecolor='white'))
    ax_mean.axhline(y=29979.2458, c="black", linestyle="--", zorder=10)
    ax_mean.text(5.6, 29600, "100ms", bbox=dict(facecolor='white'))
    save_fig(ax_mean, "mean")

    ax_max = sns.boxplot(x="SLO", y="max", hue="Shell", data=results, fliersize=1,  zorder=20, boxprops={'zorder': 20})
    ax_max.set_xlabel("SLO")
    ax_max.set_ylabel("Max. Distance to Resource Node [km]")
    ax_max.legend(loc="upper center", ncol = 4, borderaxespad=-2)
    ax_max.axhline(y=2997.92458, c="black", linestyle="--", zorder=10)
    ax_max.text(5.6, 2600, "10ms", bbox=dict(facecolor='white'))
    ax_max.axhline(y=29979.2458, c="black", linestyle="--", zorder=10)
    ax_max.text(5.6, 29600, "100ms", bbox=dict(facecolor='white'))
    save_fig(ax_max, "max")
