#
# This file is part of optimal-leo-placement
# (https://github.com/pfandzelter/optimal-leo-placement).
# Copyright (c) 2021 Ben S. Kempton, Tobias Pfandzelter.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import vtk

# used to make program faster & responsive
import threading as td

# memory aligned arrays their manipulation for Python
import numpy as np

import multiprocessing as mp
from multiprocessing.connection import Connection as MultiprocessingConnection

# Primarily using the write_gml() function...
import networkx as nx

# use to measure program performance (sim framerate)
import time

import os

import typing

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

H3_LINK_COLOR = (1.0, 0.3, 0.3)
H3_LINK_OPACITY = 1.0
H3_LINE_WIDTH = 10  # how wide to draw line in pixels

EARTH_SPHERE_POINTS = 5000  # higher = smoother earth model, slower to generate

SAT_POINT_SIZE = 9  # how big satellites are in (probably) screen pixels
GND_POINT_SIZE = 8  # how big ground points are in (probably) screen pixels

SECONDS_PER_DAY = 86400  # number of seconds per earth rotation (day)

class Animation():

    def __init__(
        self,
        total_sats: int,
        sat_positions: np.ndarray,
        current_simulation_time: int,
        earth_radius: int,
        pipe_conn: MultiprocessingConnection
    ):
        self.total_sats = total_sats
        self.sat_positions = sat_positions
        self.earth_radius = earth_radius
        self.current_simulation_time = current_simulation_time
        self.last_animate = 0
        self.links = []
        self.points = []
        self.path_links = None
        self.enable_path_calculation = False
        self.pause = False

        self.frame_count = 0
        self.incframe_count = 1

        self.make_earth_actors(self.earth_radius)

        if self.total_sats > 0:
            self.make_sats_actor(self.total_sats, self.sat_positions)

        self.make_link_actors()

        # init the 'pipe' object used for inter-process communication
        # this comes from the multiprocessing library
        self.pipe_conn = pipe_conn
        self.controlThread = td.Thread(target=self.control_thread_handler)
        self.controlThread.start()

        self.make_render_window()


###############################################################################
#                           ANIMATION FUNCTIONS                               #
###############################################################################

    """
    Like me, you might wonder what the numerous vkt calls are for.
    Answer: you need to manually configure a render pipeline for
    each object (vtk actor) in the scene.
    A typical VTK render pipeline:

    point data array   <-- set/update position data
        |
    poly data array
        |
    poly data mapper
        |
    object actor   <-- edit color/size/opacity, apply rotations/translations
        |
    vtk renderer
        |
    vkt render window
    vkt render interactor   <-- trigger events, animate
        |
    Your computer screen
    exported png files

    """

    def setup_animation(
            self,
            total_satellites: int,
            satellite_positions: np.ndarray,
            total_groundpoints: int,
            groundpoint_positions: np.ndarray,
            timestep: int = 60,
            current_simulation_time: int = 0) -> None:
        """
        Makes vtk render window, and sets up pipelines.

        Parameters
        ----------
        total_satellites : int
            The total number of satellties in the model
        satellite_positions : np.array[[('x', int32), ('y', int32), ('z', int32)]]
            Numpy array of all the satellite positions
        total_groundpoints : int
            Total number of groundpoints in the model
        groundpoint_positions : np.array[[('x', int32), ('y', int32), ('z', int32)]]
            Numpy array of all the groundpoint positions
        timestep : int
            Timestep for the simulation in seconds
        current_simulation_time : int
            current time of the simulation in seconds
        capture_images : bool
            If true, save images of the render window to file

        """


    def update_animation(self, obj: typing.Any, event: typing.Any) -> None:
        """
        This function takes in new position data and updates the render window

        Parameters
        ----------
        obj : ?
            The object that generated the event, probably vtk render window
        event : event
            The event that triggered this function
        """

        self.frame_count += 1

        # rotate earth and land
        # print("Current time: " + str(self.current_simulation_time))
        # print("Last Animate: " + str(self.last_animate))
        # print("Links: ", self.sat_positions)

        steps_to_animate = self.current_simulation_time - self.last_animate
        self.last_animate = self.current_simulation_time

        rotation_per_time_step = 360.0/(SECONDS_PER_DAY) * steps_to_animate
        self.earthActor.RotateZ(rotation_per_time_step)
        self.sphereActor.RotateZ(rotation_per_time_step)

        # grab new position data
        new_sat_positions = self.sat_positions
        # update sat points
        for i in range(self.totalSats):
            x = new_sat_positions[i]['x']
            y = new_sat_positions[i]['y']
            z = new_sat_positions[i]['z']
            self.satVtkPts.SetPoint(self.satPointIDs[i], x, y, z)
        self.satPolyData.GetPoints().Modified()

        # grab the arrays of connections
        links = self.links
        points = self.points
        maxSatIdx = self.total_sats-1

        # build a vtkPoints object from array
        self.linkPoints = vtk.vtkPoints()
        self.linkPoints.SetNumberOfPoints(len(points))
        for i in range(len(points)):
            self.linkPoints.SetPoint(i, points[i]['x'], points[i]['y'], points[i]['z'])

        # make clean line arrays
        self.islLinkLines = vtk.vtkCellArray()
        self.sglLinkLines = vtk.vtkCellArray()
        self.pathLinkLines = vtk.vtkCellArray()

        # fill isl and gsl arrays
        for i in range(len(links)):
            e1 = links[i]['node_1']
            e2 = links[i]['node_2']
            # must translate link endpoints to point names
            # if endpoint name is positive, we use it directly
            # if negative, idx = maxSatIdx-endpointname
            # **ground endpoints are always node_1**
            if e1 < 0:
                self.sglLinkLines.InsertNextCell(2)
                self.sglLinkLines.InsertCellPoint(maxSatIdx-e1)
                self.sglLinkLines.InsertCellPoint(e2)
            else:
                self.islLinkLines.InsertNextCell(2)
                self.islLinkLines.InsertCellPoint(e1)
                self.islLinkLines.InsertCellPoint(e2)

        self.sglPolyData.SetPoints(self.linkPoints)
        self.sglPolyData.SetLines(self.sglLinkLines)
        self.islPolyData.SetPoints(self.linkPoints)
        self.islPolyData.SetLines(self.islLinkLines)

        # #
        obj.GetRenderWindow().Render()

    def make_render_window(self) -> None:
        """
        Makes a render window object using vtk.

        This should not be called until all the actors are created.

        """

        # create a renderer object
        self.renderer = vtk.vtkRenderer()
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.AddRenderer(self.renderer)

        # create an interactor object, to interact with the window... duh
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.interactor.SetRenderWindow(self.renderWindow)

        # add the actor objects
        self.renderer.AddActor(self.satsActor)
        self.renderer.AddActor(self.earthActor)
        self.renderer.AddActor(self.sphereActor)

        self.renderer.AddActor(self.islActor)
        self.renderer.AddActor(self.sglActor)
        self.renderer.AddActor(self.pathActor)

        # white background, makes it easier to
        # put screenshots of animation into papers/presentations
        self.renderer.SetBackground(BACKGROUND_COLOR)

        self.interactor.Initialize()
        print('initialized interactor')

        # set up a timer to call the update function at a max rate
        # of every 7 ms (~144 hz)
        self.interactor.AddObserver('TimerEvent', self.update_animation)
        self.interactor.CreateRepeatingTimer(7)
        print('set up timer')

        # start the model
        self.renderWindow.SetSize(2048, 2048)
        self.renderWindow.Render()
        print('started render')
        self.interactor.Start()
        print('started interactor')

    def make_sats_actor(self, total_satellites: int, satellite_positions: np.ndarray) -> None:
        """
        generate the point cloud to represent satellites

        Parameters
        ----------
        total_satellites : int
            number of satellties in the simulation
        satellite_positions : np.array[[('x', int32),('y', int32),('z', int32)]]
            satellite position data, satellite "unique_id" = index number
        """

        # declare a points & cell array to hold position data
        self.satVtkPts = vtk.vtkPoints()
        self.satVtkVerts = vtk.vtkCellArray()

        # figure out the total number of sats and groundpts in the constillation
        self.totalSats = total_satellites

        # init a array for IDs
        self.satPointIDs = [None] * self.totalSats

        # initialize all the positions
        for i in range(self.totalSats):
            self.satPointIDs[i] = self.satVtkPts.InsertNextPoint(
                satellite_positions[i]['x'],
                satellite_positions[i]['y'],
                satellite_positions[i]['z'])

            self.satVtkVerts.InsertNextCell(1)
            self.satVtkVerts.InsertCellPoint(self.satPointIDs[i])

        # convert points into poly data
        # (because that's what they do in the vtk examples)
        self.satPolyData = vtk.vtkPolyData()
        self.satPolyData.SetPoints(self.satVtkPts)
        self.satPolyData.SetVerts(self.satVtkVerts)

        # create mapper object and connect to the poly data
        self.satsMapper = vtk.vtkPolyDataMapper()
        self.satsMapper.SetInputData(self.satPolyData)

        # create actor, and connect to the mapper
        # (again, its just what you do to make a vtk render pipeline)
        self.satsActor = vtk.vtkActor()
        self.satsActor.SetMapper(self.satsMapper)

        # edit appearance of satellites
        self.satsActor.GetProperty().SetOpacity(SAT_OPACITY)
        self.satsActor.GetProperty().SetColor(SAT_COLOR)
        self.satsActor.GetProperty().SetPointSize(SAT_POINT_SIZE)

    def make_link_actors(self) -> None:
        """
        generate the lines to represent links

        source:
        https://vtk.org/Wiki/VTK/Examples/Python/GeometricObjects/Display/PolyLine

        """

        # grab the arrays of connections
        links = self.links
        points = self.points
        maxSatIdx = self.total_sats-1

        # build a vtkPoints object from array
        self.linkPoints = vtk.vtkPoints()
        self.linkPoints.SetNumberOfPoints(len(points))
        for i in range(len(points)):
            self.linkPoints.SetPoint(i, points[i]['x'], points[i]['y'], points[i]['z'])

        # build a cell array to represent connectivity
        self.islLinkLines = vtk.vtkCellArray()
        self.sglLinkLines = vtk.vtkCellArray()
        for i in range(len(links)):
            e1 = links[i]['node_1']
            e2 = links[i]['node_2']
            # must translate link endpoints to point names
            # if endpoint name is positive, we use it directly
            # if negative, idx = maxSatIdx-endpointname
            # **ground endpoints are always node_1**
            if e1 < 0:
                self.sglLinkLines.InsertNextCell(2)
                self.sglLinkLines.InsertCellPoint(maxSatIdx-e1)
                self.sglLinkLines.InsertCellPoint(e2)
            else:
                self.islLinkLines.InsertNextCell(2)
                self.islLinkLines.InsertCellPoint(e1)
                self.islLinkLines.InsertCellPoint(e2)

        self.pathLinkLines = vtk.vtkCellArray()  # init, but do not fill this one

        # #

        self.islPolyData = vtk.vtkPolyData()
        self.islPolyData.SetPoints(self.linkPoints)
        self.islPolyData.SetLines(self.islLinkLines)

        self.sglPolyData = vtk.vtkPolyData()
        self.sglPolyData.SetPoints(self.linkPoints)
        self.sglPolyData.SetLines(self.sglLinkLines)

        self.pathPolyData = vtk.vtkPolyData()
        self.pathPolyData.SetPoints(self.linkPoints)
        self.pathPolyData.SetLines(self.pathLinkLines)

        # #

        self.islMapper = vtk.vtkPolyDataMapper()
        self.islMapper.SetInputData(self.islPolyData)

        self.sglMapper = vtk.vtkPolyDataMapper()
        self.sglMapper.SetInputData(self.sglPolyData)

        self.pathMapper = vtk.vtkPolyDataMapper()
        self.pathMapper.SetInputData(self.pathPolyData)

        # #

        self.islActor = vtk.vtkActor()
        self.islActor.SetMapper(self.islMapper)

        self.sglActor = vtk.vtkActor()
        self.sglActor.SetMapper(self.sglMapper)

        self.pathActor = vtk.vtkActor()
        self.pathActor.SetMapper(self.pathMapper)

        # #

        self.islActor.GetProperty().SetOpacity(ISL_LINK_OPACITY)
        self.islActor.GetProperty().SetColor(ISL_LINK_COLOR)
        self.islActor.GetProperty().SetLineWidth(ISL_LINE_WIDTH)

        self.sglActor.GetProperty().SetOpacity(SGL_LINK_OPACITY)
        self.sglActor.GetProperty().SetColor(SGL_LINK_COLOR)
        self.sglActor.GetProperty().SetLineWidth(SGL_LINE_WIDTH)

        self.pathActor.GetProperty().SetOpacity(PATH_LINK_OPACITY)
        self.pathActor.GetProperty().SetColor(PATH_LINK_COLOR)
        self.pathActor.GetProperty().SetLineWidth(PATH_LINE_WIDTH)

        # #

    def make_earth_actors(self, earth_radius: int) -> None:
        """
        generate the earth sphere, and the landmass outline

        Parameters
        ----------
        earth_radius : int
            radius of the Earth in meters

        """

        self.earthRadius = earth_radius

        # Create earth map
        # a point cloud that outlines all the earths landmass
        self.earthSource = vtk.vtkEarthSource()
        # draws as an outline of landmass, rather than fill it in
        self.earthSource.OutlineOn()

        # want this to be slightly larger than the sphere it sits on
        # so that it is not occluded by the sphere
        self.earthSource.SetRadius(self.earthRadius * 1.001)

        # controles the resolution of surface data (1 = full resolution)
        self.earthSource.SetOnRatio(1)

        # Create a mapper
        self.earthMapper = vtk.vtkPolyDataMapper()
        self.earthMapper.SetInputConnection(self.earthSource.GetOutputPort())

        # Create an actor
        self.earthActor = vtk.vtkActor()
        self.earthActor.SetMapper(self.earthMapper)

        # set color
        self.earthActor.GetProperty().SetColor(LANDMASS_OUTLINE_COLOR)
        self.earthActor.GetProperty().SetOpacity(EARTH_LAND_OPACITY)

        # make sphere data
        num_pts = EARTH_SPHERE_POINTS
        indices = np.arange(0, num_pts, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / num_pts)
        theta = np.pi * (1 + 5 ** 0.5) * indices
        x = np.cos(theta) * np.sin(phi) * self.earthRadius
        y = np.sin(theta) * np.sin(phi) * self.earthRadius
        z = np.cos(phi) * self.earthRadius

        # x,y,z is coordination of evenly distributed sphere
        # I will try to make poly data use this x,y,z
        points = vtk.vtkPoints()
        for i in range(len(x)):
            points.InsertNextPoint(x[i], y[i], z[i])

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)

        # To create surface of a sphere we need to use Delaunay triangulation
        d3D = vtk.vtkDelaunay3D()
        d3D.SetInputData(poly)  # This generates a 3D mesh

        # We need to extract the surface from the 3D mesh
        dss = vtk.vtkDataSetSurfaceFilter()
        dss.SetInputConnection(d3D.GetOutputPort())
        dss.Update()

        # Now we have our final polydata
        spherePoly = dss.GetOutput()

        # Create a mapper
        sphereMapper = vtk.vtkPolyDataMapper()
        sphereMapper.SetInputData(spherePoly)

        # Create an actor
        self.sphereActor = vtk.vtkActor()
        self.sphereActor.SetMapper(sphereMapper)

        # set color
        self.sphereActor.GetProperty().SetColor(EARTH_BASE_COLOR)
        self.sphereActor.GetProperty().SetOpacity(EARTH_OPACITY)


    def control_thread_handler(self) -> None:
        """
        Start a thread to deal with inter-process communications

        """
        while True:
            received_data = self.pipe_conn.recv()
            if type(received_data) == str:
                print(received_data)
            elif type(received_data) == list:
                command = received_data[0]
                new_data = received_data[1]
                if command == "sat_positions":
                    #print(new_data[0])
                    self.sat_positions = new_data
                if command == "links":
                    self.links = new_data
                if command == "points":
                    self.points = new_data
                if command == "total_sats":
                    self.total_sats = new_data
                if command == "enable_path_calculation":
                    self.enable_path_calculation = new_data
                if command == "path_links":
                    self.path_links = new_data
                if command == "pause":
                    self.pause = new_data
                if command == "current_simulation_time":
                    self.current_simulation_time = new_data
                    # print("updating simulation time to", self.current_simulation_time, new_data)

            else:
                print(received_data)
