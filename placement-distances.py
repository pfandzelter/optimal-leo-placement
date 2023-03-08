#
# Copyright (c) Tobias Pfandzelter. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import os
import concurrent.futures
import typing

import scipy.constants
import scipy.sparse
import scipy.sparse.csgraph
import scipy.special
import numpy as np
import pandas as pd
import tqdm
import tqdm.contrib.concurrent

import config

def prepare_calculation(name: str, planes: int, sats: int, d_intra: float, d_inter_mean: float, d_inter_max: float) -> typing.Tuple[typing.Dict[str, typing.List[int]], typing.List[int], typing.List[int]]:
    print("Calculating shell {}".format(name))
    # 1. load the placements

    considered_rnodes = set()
    rns = {}

    for t in config.SLO:
        print("Retrieving placement {}-{}".format(t["type"], t["d"]))
        t["p"] = pd.read_csv(os.path.join(config.PLACEMENTS_DIR, "{}_{}_{}.csv".format(name, t["type"], t["d"])))

        # 2. assign each node a resource node based on the placement (shortest path)

        print("Generating shortest paths")
        # graph for shortest paths
        g = np.zeros((planes * sats, planes * sats))

        # stores resource node for each node
        rn = np.zeros(planes * sats, dtype=int)

        if t["type"] == "hops":
            d_up = 1
            d_left = 1
        elif t["type"] == "mean":
            d_up = d_intra
            d_left = d_inter_mean
        else:
            d_up = d_intra
            d_left = d_inter_max

        for x in tqdm.trange(planes, desc="Adding links"):
            for y in range(sats):
                # add up link
                g[x * sats + y, x * sats + ((y + 1) % sats) ] = d_up
                # add left link
                g[x * sats + y, ((x + 1) % planes) * sats + y ] = d_left

        resource_nodes = [n["x"] * sats + n["y"]  for i, n in t["p"].iterrows()]

        dist_matrix = scipy.sparse.csgraph.dijkstra(csgraph=scipy.sparse.csr_matrix(g), indices=resource_nodes, directed=False, return_predecessors=False)

        max_best_distance = 0

        for n in tqdm.trange(planes * sats, desc="Calculating best resource node"):
            best = -1
            best_dist = np.inf
            # best_i = -1
            for i in range(len(resource_nodes)):
                if dist_matrix[i, n] < best_dist:
                    best_dist = dist_matrix[i, n]
                    best = resource_nodes[i]
                    # best_i = i

            rn[n] = int(best)
            considered_rnodes.add(int(best))

            if best_dist > max_best_distance:
                max_best_distance = best_dist

        print("Max best distance: {}".format(max_best_distance))

        rns["{}-{}".format(t["type"], t["d"])] = rn

    indices = np.zeros(planes * sats, dtype=int)
    for i, n in enumerate(considered_rnodes):
        indices[n] = i

    considered_rnodes = list(considered_rnodes)

    return rns, considered_rnodes, indices

def calculate(e: typing.Tuple[str, int, int, typing.Dict[str,  typing.List[int]], typing.List[int], typing.List[int], int, typing.Dict[str, str], str]):

    name = e[0]
    planes = e[1]
    sats = e[2]
    rns = e[3]
    considered_rnodes = e[4]
    indices = e[5]
    next_time = e[6]
    SLOs = e[7]
    results_folder = e[8]

    with open(os.path.join(results_folder, "{}-{}.csv".format(name, next_time)), "w") as f:

        f.write("type,t,n,rn,distance\n")

        # 3. for each timestep, load the shell distances

        # load csv into numpy matrix
        distances = np.loadtxt(os.path.join(config.DISTANCES_DIR, name, "{}.csv".format(next_time)), delimiter=",", skiprows=1)

        g = np.zeros((planes * sats, planes * sats))

        for d in distances:
            g[int(d[0]), int(d[1])] = d[2]

        dist_matrix = scipy.sparse.csgraph.dijkstra(csgraph=scipy.sparse.csr_matrix(g), indices=considered_rnodes, directed=False, return_predecessors=False)

        # 4. calculate the distance between each node and the resource node
        for t in SLOs:
            for n in range(planes * sats):
                r = indices[rns["{}-{}".format(t["type"], t["d"])][n]]
                f.write("{},{},{},{},{}\n".format(t["type"], t["d"], n, rns["{}-{}".format(t["type"], t["d"])][n], dist_matrix[r, n]))


if __name__ == "__main__":
    for s in config.SHELLS:
        s["rns"], s["considered_rnodes"], s["indices"] = prepare_calculation(s["name"], s["planes"], s["sats"], s["D_M"], s["D_N_mean"], s["D_N_max"])

    print("Going through simulation steps")

    tqdm.contrib.concurrent.process_map(calculate, [(s["name"], s["planes"], s["sats"], s["rns"], s["considered_rnodes"], s["indices"], t, config.SLO, config.PLACEMENTS_DIR) for s in config.SHELLS for t in range(0, config.STEPS, config.INTERVAL)], chunksize=10, max_workers=os.cpu_count())
