import tqdm
import os
import sys
import concurrent.futures

import config
from simulation.simulation import Simulation

sys.path.append(os.path.abspath(os.getcwd()))

def run_simulation(steps: int, interval: float, planes: int, nodes: int, inc: float, altitude: int, name: str, results_folder: str):

    # setup simulation
    s = Simulation(planes=planes, nodes_per_plane=nodes, inclination=inc, semi_major_axis=int(altitude + config.EARTH_RADIUS)*1000, earth_radius=int(config.EARTH_RADIUS * 1000), model=config.MODEL, animate=config.ANIMATE, report_status=config.DEBUG)
    # for each timestep, run simulation

    total_steps = int(steps/interval)
    for step in tqdm.trange(total_steps, desc="simulating {}".format(name)):
    # for step in range(total_steps):
        next_time = step*interval

        with open(os.path.join(results_folder, name, "{}.csv".format(next_time)), "w") as f:
            f.write("a,b,distance\n")

            s.update_model(next_time, result_file=f)

    if s.animation is not None:
        s.animation.terminate()

    s.terminate()

if __name__ == "__main__":
    os.makedirs(config.DISTANCES_DIR, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for s in config.SHELLS:
            PLANES = s["planes"]
            # Number of nodes/plane
            NODES = s["sats"]

            # Plane inclination (deg)
            INC = s["inc"]

            # Orbit Altitude (Km)
            ALTITUDE = s["altitude"]

            NAME = s["name"]

            executor.submit(run_simulation, config.STEPS, config.INTERVAL, int(PLANES), int(NODES), float(INC), int(ALTITUDE), NAME, config.DISTANCES_DIR)
