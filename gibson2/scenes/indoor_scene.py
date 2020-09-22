import logging
import pickle
import networkx as nx
import cv2
from PIL import Image
import numpy as np
from gibson2.objects.articulated_object import ArticulatedObject, URDFObject
from gibson2.utils.utils import l2_distance, get_transform_from_xyz_rpy, quatXYZWFromRotMat
from gibson2.utils.assets_utils import get_scene_path, get_texture_file, get_ig_scene_path
import pybullet_data
import pybullet as p
import os
from gibson2.scenes.scene_base import Scene


class IndoorScene(Scene):
    """
    Indoor scene class for Gibson and iGibson.
    Contains the functionalities for navigation such as shortest path computation
    """

    def __init__(self,
                 scene_id,
                 trav_map_resolution=0.1,
                 trav_map_erosion=2,
                 build_graph=True,
                 num_waypoints=10,
                 waypoint_resolution=0.2,
                 pybullet_load_texture=False,
                 ):
        """
        Load an indoor scene and compute traversability

        :param scene_id: Scene id
        :param trav_map_resolution: traversability map resolution
        :param trav_map_erosion: erosion radius of traversability areas, should be robot footprint radius
        :param build_graph: build connectivity graph
        :param num_waypoints: number of way points returned
        :param waypoint_resolution: resolution of adjacent way points
        :param pybullet_load_texture: whether to load texture into pybullet. This is for debugging purpose only and
        does not affect robot's observations
        """
        super().__init__()
        logging.info("IndoorScene model: {}".format(scene_id))
        self.scene_id = scene_id
        self.trav_map_default_resolution = 0.01  # each pixel represents 0.01m
        self.trav_map_resolution = trav_map_resolution
        self.trav_map_original_size = None
        self.trav_map_size = None
        self.trav_map_erosion = trav_map_erosion
        self.build_graph = build_graph
        self.num_waypoints = num_waypoints
        self.waypoint_interval = int(waypoint_resolution / trav_map_resolution)
        self.mesh_body_id = None
        self.pybullet_load_texture = pybullet_load_texture
        self.floor_heights = [0.0]

    def load_trav_map(self, maps_path):
        """
        Loads the traversability maps for all floors
        :param maps_path: String with the path to the folder containing the traversability maps
        :return: None
        """
        self.floor_map = []
        self.floor_graph = []
        for f in range(len(self.floor_heights)):
            trav_map = np.array(Image.open(
                os.path.join(maps_path,
                             'floor_trav_{}.png'.format(f))
            ))
            obstacle_map = np.array(Image.open(
                os.path.join(maps_path,
                             'floor_{}.png'.format(f))
            ))
            if self.trav_map_original_size is None:
                height, width = trav_map.shape
                assert height == width, 'trav map is not a square'
                self.trav_map_original_size = height
                self.trav_map_size = int(self.trav_map_original_size *
                                         self.trav_map_default_resolution /
                                         self.trav_map_resolution)
            trav_map[obstacle_map == 0] = 0
            trav_map = cv2.resize(trav_map, (self.trav_map_size, self.trav_map_size))
            trav_map = cv2.erode(trav_map, np.ones((self.trav_map_erosion, self.trav_map_erosion)))
            trav_map[trav_map < 255] = 0

            if self.build_graph:
                graph_file = os.path.join(maps_path, 'floor_trav_{}.p'.format(f))
                if os.path.isfile(graph_file):
                    logging.info("Loading traversable graph")
                    with open(graph_file, 'rb') as pfile:
                        g = pickle.load(pfile)
                else:
                    logging.info("Building traversable graph")
                    g = nx.Graph()
                    for i in range(self.trav_map_size):
                        for j in range(self.trav_map_size):
                            if trav_map[i, j] > 0:
                                g.add_node((i, j))
                                # 8-connected graph
                                neighbors = [
                                    (i - 1, j - 1), (i, j - 1), (i + 1, j - 1), (i - 1, j)]
                                for n in neighbors:
                                    if 0 <= n[0] < self.trav_map_size and 0 <= n[1] < self.trav_map_size and trav_map[n[0], n[1]] > 0:
                                        g.add_edge(
                                            n, (i, j), weight=l2_distance(n, (i, j)))

                    # only take the largest connected component
                    largest_cc = max(nx.connected_components(g), key=len)
                    g = g.subgraph(largest_cc).copy()
                    with open(graph_file, 'wb') as pfile:
                        pickle.dump(g, pfile, protocol=pickle.HIGHEST_PROTOCOL)

                self.floor_graph.append(g)
                # update trav_map accordingly
                trav_map[:, :] = 0
                for node in g.nodes:
                    trav_map[node[0], node[1]] = 255

            self.floor_map.append(trav_map)

    def get_random_point(self, floor=None, random_height=False):
        if floor is None:
            floor = self.get_random_floor()
        trav = self.floor_map[floor]
        trav_space = np.where(trav == 255)
        idx = np.random.randint(0, high=trav_space[0].shape[0])
        xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
        x, y = self.map_to_world(xy_map)
        z = self.floor_heights[floor]
        if random_height:
            z += np.random.uniform(0.4, 0.8)
        return floor, np.array([x, y, z])

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
        return np.flip((xy / self.trav_map_resolution + self.trav_map_size / 2.0)).astype(np.int)

    def has_node(self, floor, world_xy):
        map_xy = tuple(self.world_to_map(world_xy))
        g = self.floor_graph[floor]
        return g.has_node(map_xy)

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        assert self.build_graph, 'cannot get shortest path without building the graph'
        source_map = tuple(self.world_to_map(source_world))
        target_map = tuple(self.world_to_map(target_world))

        g = self.floor_graph[floor]

        if not g.has_node(target_map):
            nodes = np.array(g.nodes)
            closest_node = tuple(
                nodes[np.argmin(np.linalg.norm(nodes - target_map, axis=1))])
            g.add_edge(closest_node, target_map,
                       weight=l2_distance(closest_node, target_map))

        if not g.has_node(source_map):
            nodes = np.array(g.nodes)
            closest_node = tuple(
                nodes[np.argmin(np.linalg.norm(nodes - source_map, axis=1))])
            g.add_edge(closest_node, source_map,
                       weight=l2_distance(closest_node, source_map))

        path_map = np.array(nx.astar_path(
            g, source_map, target_map, heuristic=l2_distance))

        path_world = self.map_to_world(path_map)
        geodesic_distance = np.sum(np.linalg.norm(
            path_world[1:] - path_world[:-1], axis=1))
        path_world = path_world[::self.waypoint_interval]

        if not entire_path:
            path_world = path_world[:self.num_waypoints]
            num_remaining_waypoints = self.num_waypoints - path_world.shape[0]
            if num_remaining_waypoints > 0:
                remaining_waypoints = np.tile(
                    target_world, (num_remaining_waypoints, 1))
                path_world = np.concatenate(
                    (path_world, remaining_waypoints), axis=0)

        return path_world, geodesic_distance
