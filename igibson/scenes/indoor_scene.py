import logging
import os
import pickle
import sys
from abc import ABCMeta

import cv2
import networkx as nx
import numpy as np
from future.utils import with_metaclass
from PIL import Image

from igibson.scenes.scene_base import Scene
from igibson.utils.utils import l2_distance

log = logging.getLogger(__name__)


class IndoorScene(with_metaclass(ABCMeta, Scene)):
    """
    Indoor scene class for Gibson and iGibson.
    Contains the functionalities for navigation such as shortest path computation
    """

    def __init__(
        self,
        scene_id,
        trav_map_resolution=0.1,
        trav_map_erosion=2,
        trav_map_type="with_obj",
        build_graph=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
    ):
        """
        Load an indoor scene and compute traversability

        :param scene_id: Scene id
        :param trav_map_resolution: traversability map resolution
        :param trav_map_erosion: erosion radius of traversability areas, should be robot footprint radius
        :param trav_map_type: type of traversability map, with_obj | no_obj
        :param build_graph: build connectivity graph
        :param num_waypoints: number of way points returned
        :param waypoint_resolution: resolution of adjacent way points
        """
        super(IndoorScene, self).__init__()
        log.debug("IndoorScene model: {}".format(scene_id))
        self.scene_id = scene_id
        self.trav_map_default_resolution = 0.01  # each pixel represents 0.01m
        self.trav_map_resolution = trav_map_resolution
        self.trav_map_original_size = None
        self.trav_map_size = None
        self.trav_map_erosion = trav_map_erosion
        self.trav_map_type = trav_map_type
        self.build_graph = build_graph
        self.num_waypoints = num_waypoints
        self.waypoint_interval = int(waypoint_resolution / trav_map_resolution)
        self.mesh_body_id = None
        self.floor_heights = [0.0]

    def load_trav_map(self, maps_path):
        """
        Loads the traversability maps for all floors

        :param maps_path: String with the path to the folder containing the traversability maps
        """
        if not os.path.exists(maps_path):
            log.warning("trav map does not exist: {}".format(maps_path))
            return

        self.floor_map = []
        self.floor_graph = []
        for floor in range(len(self.floor_heights)):
            if self.trav_map_type == "with_obj":
<<<<<<< HEAD
                # for this project Sonicverse, we will not render any doors in the trav map
                trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_no_door_{}.png".format(floor))))
                # obstacle_map = np.array(Image.open(os.path.join(maps_path, "floor_{}.png".format(floor))))
            else:
                trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_no_obj_{}.png".format(floor))))
                # obstacle_map = np.array(Image.open(os.path.join(maps_path, "floor_no_obj_{}.png".format(floor))))
=======
                trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_{}.png".format(floor))))
                obstacle_map = np.array(Image.open(os.path.join(maps_path, "floor_{}.png".format(floor))))
            else:
                trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_no_obj_{}.png".format(floor))))
                obstacle_map = np.array(Image.open(os.path.join(maps_path, "floor_no_obj_{}.png".format(floor))))
>>>>>>> ddbfc8be187008cd173688c95cad12dc1bbf7c9b

            # If we do not initialize the original size of the traversability map, we obtain it from the image
            # Then, we compute the final map size as the factor of scaling (default_resolution/resolution) times the
            # original map size
            if self.trav_map_original_size is None:
                height, width = trav_map.shape
                assert height == width, "trav map is not a square"
                self.trav_map_original_size = height
                self.trav_map_size = int(
                    self.trav_map_original_size * self.trav_map_default_resolution / self.trav_map_resolution
                )

            # Here it looks like we do not "care" about the traversability map: wherever the obstacle map is 0, we set
            # the traversability map also to 0
<<<<<<< HEAD
            # trav_map[obstacle_map == 0] = 0
=======
            trav_map[obstacle_map == 0] = 0
>>>>>>> ddbfc8be187008cd173688c95cad12dc1bbf7c9b

            # We resize the traversability map to the new size computed before
            trav_map = cv2.resize(trav_map, (self.trav_map_size, self.trav_map_size))

            # We then erode the image. This is needed because the code that computes shortest path uses the global map
            # and a point robot
<<<<<<< HEAD
            
=======
>>>>>>> ddbfc8be187008cd173688c95cad12dc1bbf7c9b
            if self.trav_map_erosion != 0:
                trav_map = cv2.erode(trav_map, np.ones((self.trav_map_erosion, self.trav_map_erosion)))

            # We make the pixels of the image to be either 0 or 255
            trav_map[trav_map < 255] = 0
<<<<<<< HEAD
            cv2.imwrite("/viscam/u/li2053/iGibson-dev/igibson/agents/savi_rt/trav_map_1.png", trav_map.astype(np.uint8))
            # We search for the largest connected areas
            if self.build_graph:
                self.build_trav_graph(maps_path, floor, trav_map)
            cv2.imwrite("/viscam/u/li2053/iGibson-dev/igibson/agents/savi_rt/trav_map_2.png", trav_map.astype(np.uint8))
=======
#             cv2.imwrite("trav_map_1.png", trav_map.astype(np.uint8))
            # We search for the largest connected areas
            if self.build_graph:
                self.build_trav_graph(maps_path, floor, trav_map)
#             cv2.imwrite("trav_map_2.png", trav_map.astype(np.uint8))
>>>>>>> ddbfc8be187008cd173688c95cad12dc1bbf7c9b
            self.floor_map.append(trav_map)

    # TODO: refactor into C++ for speedup
    def build_trav_graph(self, maps_path, floor, trav_map):
        """
        Build traversibility graph and only take the largest connected component

        :param maps_path: String with the path to the folder containing the traversability maps
        :param floor: floor number
        :param trav_map: traversability map
        """
<<<<<<< HEAD
        # graph_file = os.path.join(
        #     maps_path, "floor_trav_{}_py{}{}.p".format(floor, sys.version_info.major, sys.version_info.minor)
        # )
        # if os.path.isfile(graph_file):
        #     log.debug("Loading traversable graph")
        #     with open(graph_file, "rb") as pfile:
        #         g = pickle.load(pfile)
        # else:
        #     log.debug("Building traversable graph")
        #     g = nx.Graph()
        #     for i in range(self.trav_map_size):
        #         for j in range(self.trav_map_size):
        #             if trav_map[i, j] == 0:
        #                 continue
        #             g.add_node((i, j))
        #             # 8-connected graph
        #             neighbors = [(i - 1, j - 1), (i, j - 1), (i + 1, j - 1), (i - 1, j)]
        #             for n in neighbors:
        #                 if (
        #                     0 <= n[0] < self.trav_map_size
        #                     and 0 <= n[1] < self.trav_map_size
        #                     and trav_map[n[0], n[1]] > 0
        #                 ):
        #                     g.add_edge(n, (i, j), weight=l2_distance(n, (i, j)))

        #     # only take the largest connected component
        #     largest_cc = max(nx.connected_components(g), key=len)
        #     g = g.subgraph(largest_cc).copy()
        #     with open(graph_file, "wb") as pfile:
        #         pickle.dump(g, pfile)

        log.debug("Building traversable graph")
        g = nx.Graph()
        for i in range(self.trav_map_size):
            for j in range(self.trav_map_size):
                if trav_map[i, j] == 0:
                    continue
                g.add_node((i, j))
                # 8-connected graph
                neighbors = [(i - 1, j - 1), (i, j - 1), (i + 1, j - 1), (i - 1, j)]
                for n in neighbors:
                    if 0 <= n[0] < self.trav_map_size and 0 <= n[1] < self.trav_map_size and trav_map[n[0], n[1]] > 0:
                        g.add_edge(n, (i, j), weight=l2_distance(n, (i, j)))

        # only take the largest connected component
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()
=======
        graph_file = os.path.join(
            maps_path, "floor_trav_{}_py{}{}.p".format(floor, sys.version_info.major, sys.version_info.minor)
        )
        if os.path.isfile(graph_file):
            log.debug("Loading traversable graph")
            with open(graph_file, "rb") as pfile:
                g = pickle.load(pfile)
        else:
            log.debug("Building traversable graph")
            g = nx.Graph()
            for i in range(self.trav_map_size):
                for j in range(self.trav_map_size):
                    if trav_map[i, j] == 0:
                        continue
                    g.add_node((i, j))
                    # 8-connected graph
                    neighbors = [(i - 1, j - 1), (i, j - 1), (i + 1, j - 1), (i - 1, j)]
                    for n in neighbors:
                        if (
                            0 <= n[0] < self.trav_map_size
                            and 0 <= n[1] < self.trav_map_size
                            and trav_map[n[0], n[1]] > 0
                        ):
                            g.add_edge(n, (i, j), weight=l2_distance(n, (i, j)))

            # only take the largest connected component
            largest_cc = max(nx.connected_components(g), key=len)
            g = g.subgraph(largest_cc).copy()
            with open(graph_file, "wb") as pfile:
                pickle.dump(g, pfile)
>>>>>>> ddbfc8be187008cd173688c95cad12dc1bbf7c9b

        self.floor_graph.append(g)

        # update trav_map accordingly
        # This overwrites the traversability map loaded before
        # It sets everything to zero, then only sets to one the points where we have graph nodes
        # Dangerous! if the traversability graph is not computed from the loaded map but from a file, it could overwrite
        # it silently.
        trav_map[:, :] = 0
        for node in g.nodes:
            trav_map[node[0], node[1]] = 255

    def get_random_point(self, floor=None):
        """
        Sample a random point on the given floor number. If not given, sample a random floor number.

        :param floor: floor number
        :return floor: floor number
        :return point: randomly sampled point in [x, y, z]
        """
        if floor is None:
            floor = self.get_random_floor()
        trav = self.floor_map[floor]
        trav_space = np.where(trav == 255)
        idx = np.random.randint(0, high=trav_space[0].shape[0])
        xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
        x, y = self.map_to_world(xy_map)
        z = self.floor_heights[floor]
        return floor, np.array([x, y, z])

    def get_points_grid(self, num_points):
        """
        Get a grid of size num_points x num_points consisting of points in the traversible space

        :param num_points: number of points along each axis
        :return points_grid: sampled points in [xyz] mapped to their corresponding floors
        """
        points_grid = {floor:[] for floor in range(len(self.floor_heights))}
        for floor in range(len(self.floor_heights)):
            z = self.floor_heights[floor]

            trav = self.floor_map[floor]
            trav_space = np.where(trav == 255)

            x_step = int(trav_space[0].shape[0] / (num_points-1))
            y_step = int(trav_space[1].shape[0] / (num_points-1))

            for i in range(0, trav_space[0].shape[0], x_step):
                for j in range(0, trav_space[1].shape[0], y_step):
                    xy_map = np.array([trav_space[0][i], trav_space[1][j]])
                    x, y = self.map_to_world(xy_map)
                    points_grid[floor].append(np.array([x, y, z]))
        return points_grid

    def map_to_world(self, xy):
        """
        Transforms a 2D point in map reference frame into world (simulator) reference frame

        :param xy: 2D location in map reference frame (image)
        :return: 2D location in world reference frame (metric)
        """
        axis = 0 if len(xy.shape) == 1 else 1
        return np.flip((xy - self.trav_map_size / 2.0) * self.trav_map_resolution, axis=axis)

    def world_to_map(self, xy):
        """
        Transforms a 2D point in world (simulator) reference frame into map reference frame

        :param xy: 2D location in world reference frame (metric)
        :return: 2D location in map reference frame (image)
        """
        return np.flip((np.array(xy) / self.trav_map_resolution + self.trav_map_size / 2.0)).astype(np.int)

    def has_node(self, floor, world_xy):
        """
        Return whether the traversability graph contains a point

        :param floor: floor number
        :param world_xy: 2D location in world reference frame (metric)
        """
        map_xy = tuple(self.world_to_map(world_xy))
        g = self.floor_graph[floor]
        return g.has_node(map_xy)

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        """
        Get the shortest path from one point to another point.
        If any of the given point is not in the graph, add it to the graph and
        create an edge between it to its closest node.

        :param floor: floor number
        :param source_world: 2D source location in world reference frame (metric)
        :param target_world: 2D target location in world reference frame (metric)
        :param entire_path: whether to return the entire path
        """
        # call every step to calculate the geodesic potential
        assert self.build_graph, "cannot get shortest path without building the graph"
        source_map = tuple(self.world_to_map(source_world))
        target_map = tuple(self.world_to_map(target_world))

        g = self.floor_graph[floor]

        if not g.has_node(target_map):
            nodes = np.array(g.nodes)
            closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - target_map, axis=1))])
            g.add_edge(closest_node, target_map, weight=l2_distance(closest_node, target_map))

        if not g.has_node(source_map):
            nodes = np.array(g.nodes)
            closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - source_map, axis=1))])
            g.add_edge(closest_node, source_map, weight=l2_distance(closest_node, source_map))

        path_map = np.array(nx.astar_path(g, source_map, target_map, heuristic=l2_distance))

        path_world = self.map_to_world(path_map)
        geodesic_distance = np.sum(np.linalg.norm(path_world[1:] - path_world[:-1], axis=1))
        path_world = path_world[:: self.waypoint_interval]

        if not entire_path:
            path_world = path_world[: self.num_waypoints]
            num_remaining_waypoints = self.num_waypoints - path_world.shape[0]
            if num_remaining_waypoints > 0:
                remaining_waypoints = np.tile(target_world, (num_remaining_waypoints, 1))
                path_world = np.concatenate((path_world, remaining_waypoints), axis=0)

        return path_world, geodesic_distance
