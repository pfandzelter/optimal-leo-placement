import os
import numpy as np
import scipy.constants
import scipy.special

# whether to show animations of simulated constellations
ANIMATE = False

# whether to print debug information
DEBUG = False

# number of columns in the output terminal
TERM_SIZE = 80

# radius of earth in km
EARTH_RADIUS = 6371.0

# model to use for satellite orbits
# can be SGP4 or Kepler
MODEL = "SGP4"

# simulation interval in seconds
INTERVAL = 1

# total length of the simulation in seconds
STEPS = 86400

# speed of light in km/s
C = scipy.constants.speed_of_light / 1000.0

# output folders
__root = os.path.abspath(os.path.dirname(__file__)) if __file__ else "."
PLACEMENTS_DIR = os.path.join(__root, "placements")
os.makedirs(PLACEMENTS_DIR, exist_ok=True)
DISTANCES_DIR = os.path.join(__root, "distances-results")
os.makedirs(DISTANCES_DIR, exist_ok=True)
PLACEMENT_DISTANCES_DIR = os.path.join(__root, "placement-distances")
os.makedirs(PLACEMENT_DISTANCES_DIR, exist_ok=True)
GRAPHS_DIR = os.path.join(__root, "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(__root, "results.csv")

# constellation shells to consider
SHELLS = [
    # Starlink Shell A
    {
        "name": "st1",
        "planes": 72,
        "sats": 22,
        "altitude": 550,
        "inc": 53.0,
    },
    # Starlink Shell B
    {
        "name": "st2",
        "planes": 5,
        "sats": 75,
        "altitude": 1275,
        "inc": 81.0,
    },
    # Kuiper Shell A
    {
        "name": "ku1",
        "planes": 34,
        "sats": 34,
        "altitude": 630,
        "inc": 51.9,
    },
    # Kuiper Shell B
    {
        "name": "ku2",
        "planes": 28,
        "sats": 28,
        "altitude": 590,
        "inc": 33.0,
    },
]

# SLOs to consider
# type can be hops (discrete distances), mean (mean distances), or max (maximum distances)
SLO = [
    {
        "type": "hops",
        "t": 1,
    },
    {
        "type": "hops",
        "t": 4,
    },
    {
        "type": "mean",
        # 10ms * c, convert ms to s and m to km
        "t": 10 * 0.001 * C,
    },
    {
        "type": "max",
        # 10ms * c, convert ms to s and m to km
        "t": 10 * 0.001 * C,
    },
        {
        "type": "mean",
        # 100ms * c, convert ms to s and m to km
        "t": 100 * 0.001 * C,
    },
    {
        "type": "max",
        # 100ms * c, convert ms to s and m to km
        "t": 100 * 0.001 * C,
    }
]

# generate the distances
for s in SHELLS:
    if DEBUG:
        print("Calculating distances for {}".format(s["name"]))

    # intra-plane distance
    s["d_intra"] = (EARTH_RADIUS + s["altitude"]) * np.sqrt(2 * (1 - np.cos((2*np.pi) / s["sats"])))
    if DEBUG:
        print("Intra-plane distance: {}".format(s["d_intra"]))
    # max inter distance
    s["d_inter_max"] = (EARTH_RADIUS + s["altitude"]) * np.sqrt(2 * (1 - np.cos((2*np.pi) / s["planes"])))
    if DEBUG:
        print("Max inter-plane distance: {}".format(s["d_inter_max"]))

    # mean inter distance
    s["d_inter_mean"] = (2 / np.pi) * (EARTH_RADIUS + s["altitude"]) * np.sqrt(2 * (1 - np.cos((2*np.pi) / s["planes"]))) * scipy.special.ellipe(1 - (np.cos(np.deg2rad(s["inc"])))**2)
    if DEBUG:
        print("Mean inter-plane distance: {}".format(s["d_inter_mean"]))
