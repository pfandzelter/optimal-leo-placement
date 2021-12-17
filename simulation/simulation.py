# satellite network model, high level simulation control
# called from a Qt5 UI

# Author: Ben S. Kempton, Tobias Pfandzelter

# used to run animation in a different process
import multiprocessing as mp

# custom classes
from .constellation import Constellation

# use to measure program performance (sim framerate)
import time

import typing

# try to import numba funcs
try:
    import numba
    #import numba_funcs as nf
    USING_NUMBA = True
except ModuleNotFoundError:
    USING_NUMBA = False
    print("you probably do not have numba installed...")
    print("reverting to non-numba mode")


###############################################################################
#                               GLOBAL VARS                                   #
###############################################################################

LANDMASS_OUTLINE_COLOR = (0.0, 0.0, 0.0)  # black, best contrast
EARTH_LAND_OPACITY = 1.0

EARTH_BASE_COLOR = (0.6, 0.6, 0.8)  # light blue, like water!
EARTH_OPACITY = 1.0

BACKGROUND_COLOR = (1.0, 1.0, 1.0)  # white

SAT_COLOR = (1.0, 0.0, 0.0)  # red, color of satellites
SAT_OPACITY = 1.0

GND_COLOR = (0.0, 1.0, 0.0)  # green, color of groundstations
GND_OPACITY = 1.0

ISL_LINK_COLOR = (0.9, 0.5, 0.1)  # yellow-brown, satellite-satellite links
ISL_LINK_OPACITY = 1.0
ISL_LINE_WIDTH = 3  # how wide to draw line in pixels

SGL_LINK_COLOR = (0.5, 0.9, 0.5)  # greenish? satellite-groundstation links
SGL_LINK_OPACITY = 0.75
SGL_LINE_WIDTH = 2  # how wide to draw line in pixels

PATH_LINK_COLOR = (0.8, 0.2, 0.8)  # purpleish? path links
PATH_LINK_OPACITY = 0.7
PATH_LINE_WIDTH = 13  # how wide to draw line in pixels

EARTH_SPHERE_POINTS = 5000  # higher = smoother earth model, slower to generate

SAT_POINT_SIZE = 9  # how big satellites are in (probably) screen pixels
GND_POINT_SIZE = 8  # how big ground points are in (probably) screen pixels

SECONDS_PER_DAY = 86400  # number of seconds per earth rotation (day)


###############################################################################
#                             SIMULATION CONTROL                              #
###############################################################################


class Simulation():

    def __init__(
            self,
            planes: int = 1,
            nodes_per_plane: int = 1,
            inclination: float = 70.0,
            semi_major_axis: int = 6472000,
            earth_radius: int = 6371000,
            model: str = "Kepler",
            animate: bool = True,
            report_status: bool = False):

        # constillation structure information
        self.num_planes = planes
        self.num_nodes_per_plane = nodes_per_plane
        self.plane_inclination = inclination
        self.semi_major_axis = semi_major_axis
        self.min_communications_altitude = 100000

        # control flags
        self.animate = animate
        self.report_status = report_status

        # timing control
        self.current_simulation_time = 0.0
        self.pause = False

        self.animation: typing.Optional[mp.Process] = None

        if model == "Kepler":
            use_SGP4 = False
        elif model == "SGP4":
            use_SGP4 = True
        else:
            raise ValueError("invalid model: " + model)

        # init the Constellation model
        self.model = Constellation(
            planes=self.num_planes,
            nodes_per_plane=self.num_nodes_per_plane,
            inclination=self.plane_inclination,
            semi_major_axis=self.semi_major_axis,
            min_communications_altitude=self.min_communications_altitude,
            use_SGP4=use_SGP4,
            earth_radius=earth_radius)

        # init the network design
        self.initialize_network_design()

        # so, after much effort it appears that I cannot control an
        # interactive vtk window externally. Therefore when running
        # with an animation, the animation class will have to drive
        # the simulation using an internal timer...
        if self.animate:

            from .animation import Animation

            parent_conn, child_conn = mp.Pipe()

            kw = {
                "total_sats": self.model.total_sats,
                "sat_positions": self.model.get_array_of_sat_positions(),
                "current_simulation_time": self.current_simulation_time,
                "earth_radius": earth_radius,
                "pipe_conn": child_conn,
            }

            self.animation = mp.Process(target=Animation, kwargs=kw)
            self.animation.start()

            time.sleep(10)
            self.pipe_conn = parent_conn

    def terminate(self) -> None:
        if self.animation is not None:
            self.animation.join()
            self.animation.close()

    def initialize_network_design(self) -> None:
        if self.report_status:
            print("initalizing network design... ")

        self.max_isl_distance = self.model.calculate_max_ISL_distance(
            self.min_communications_altitude)

        if self.report_status:
            print("maxIsl: ", self.max_isl_distance)

        self.model.init_plus_grid_links()

        if self.report_status:
            print("done initalizing")

    def update_model(self, new_time: float, result_file: typing.TextIO) -> None:
        """
        Update the model with a new time & recalculate links

        Function behaves differently depending on wether animate is true or not.
        If true, this func will be called from the updateAnimation() func
        If False, this will be called in a loop until some desired runtime is reached

        """

        time_1 = time.time()

        # grab initial time
        self.model.set_constellation_time(time=new_time)

        time_2 = time.time()

        self.model.update_plus_grid_links(max_isl_range=self.max_isl_distance)

        time_3 = time.time()

        links = self.model.get_array_of_links()
        if result_file is not None:
            for l in links:
                if l["active"]:
                    # let's only care about one node for now
                    #if l["node_1"] == 0 or l["node_2"] == 0:
                    # result_file.write(str(new_time))
                    # result_file.write(",")
                    result_file.write(str(l["node_1"]))
                    result_file.write(",")
                    result_file.write(str(l["node_2"]))
                    result_file.write(",")
                    result_file.write(str(l["distance"]))
                    # result_file.write(",")
                    # result_file.write(str(self.plane_inclination))
                    result_file.write("\n")

        time_4 = time.time()

        if self.animate:
            self.pipe_conn.send(["sat_positions", self.model.get_array_of_sat_positions()])
            self.pipe_conn.send(["links", links])
            self.pipe_conn.send(["points",self.model.get_array_of_node_positions()])
            self.pipe_conn.send(["total_sats", self.model.total_sats])
            self.pipe_conn.send(["pause", self.pause])
            self.pipe_conn.send(["current_simulation_time", new_time])

        self.current_simulation_time = new_time

        if self.animate:
            self.pipe_conn.send(["current_simulation_time", self.current_simulation_time])

        time_5 = time.time()

        if self.report_status:
            print("set constellation time:", (time_2 - time_1))
            print("update links:", (time_3 - time_2))
            print("write to file:", (time_4 - time_3))
            print("rest:", (time_5 - time_4))
