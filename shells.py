import numpy as np
import os
import typing
import matplotlib.pyplot as plt
import seaborn as sns

import placement
import config

def save_placement(name: str, p: typing.List[typing.Tuple[int, int]]) -> None:
    with open(os.path.join(config.PLACEMENTS_DIR, "{}.csv".format(name)), "w") as f:
        f.write("x,y\n")
        for x, y in p:
            f.write("{},{}\n".format(x, y))

def save_placement_picture(name: str, k1: int, k2: int, d1: typing.Union[int, float], d2: typing.Union[int, float], t: typing.Union[int, float], p: typing.List[typing.Tuple[int, int]]) -> None:

    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca()
    ax.set_xticks(np.arange(0, k1, 1))
    ax.set_xlim(0, k1-1)
    ax.set_yticks(np.arange(0, k2, 1))
    ax.set_ylim(0, k2-1)
    ax.grid()

    pal = sns.color_palette(n_colors=len(p))

    n = np.array(p)

    covered = set()

    for i in range(0, len(n)):
        rn = n[i]
        color = pal[i]

        inrangex = []
        inrangey = []

        for bx in range(0, k1):
            for by in range(0, k2):
                if placement.weighted_lee(rn, (bx, by), k1, k2, d1, d2) <= t:
                    inrangex.append(bx)
                    inrangey.append(by)
                    covered.add((bx, by))

        ax.scatter(inrangex, inrangey, color=color, marker="s", s=100, zorder=10)

    uncovered = set([ (bx, by) for bx in range(0, k1) for by in range(0, k2) ])

    uncovered = uncovered.difference(covered)
    # print(uncovered)
    uncovered = list(uncovered)
    # print(uncovered)
    if len(uncovered) > 0:
        uncovered = np.array(uncovered)
        ax.scatter(uncovered[:,0], uncovered[:,1],  color="red", marker="s", s=100, zorder=10)

    ax.scatter(n[:,0], n[:,1], axes=ax, s=200, color="black", zorder=12)
    fig.savefig(os.path.join(config.PLACEMENTS_DIR, "{}.png".format(name)), format='png')
    plt.close(fig)

# for each shell and slo, generate a placement, save it to a file, save it as a picture
for s in config.SHELLS:
    for slo in config.SLO:
        print("=" * config.TERM_SIZE)
        print("Generating {}-{} for {}".format(slo["t"], slo["type"], s["name"]))
        print("")
        # generate the placement
        k1 = s["planes"]
        k2 = s["sats"]
        if slo["type"] == "hops":
            d1 = 1
            d2 = 1
        elif slo["type"] == "mean":
            d1 = s["d_inter_mean"]
            d2 = s["d_intra"]
        elif slo["type"] == "max":
            d1 = s["d_inter_max"]
            d2 = s["d_intra"]
        else:
            raise ValueError("SLO type not supported")

        p, eps = placement.placement(k1, k2, d1, d2, slo["t"], debug=config.DEBUG)

        print("k1: {}\nk2: {}\nd1: {}\nd2: {}\nt: {}\n".format(k1, k2, d1, d2, slo["t"], eps))

        print("Number of resource nodes: {}".format(len(p)))
        print("Epsilon: {}".format(eps))

        # save the placement to a file
        name = "{}_{}_{}".format(s["name"], slo["type"], slo["t"])

        save_placement(name, p)

        # save the placement as a picture
        save_placement_picture(name, k1, k2, d1, d2, slo["t"], p)
        print("=" * config.TERM_SIZE)