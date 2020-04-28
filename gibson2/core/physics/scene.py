import pybullet as p
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet_data
from gibson2.utils.assets_utils import get_model_path, get_texture_file
from gibson2.utils.utils import l2_distance
from gibson2.core.physics.interactive_objects import InteractiveObj

import numpy as np
from PIL import Image
import cv2
import networkx as nx
import pickle
import logging

class Scene:
    def load(self):
        raise NotImplementedError()

class EmptyScene(Scene):
    """
    A empty scene for debugging
    """
    def load(self):
        self.build_graph = False
        self.is_interactive = False
        planeName = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.ground = p.loadMJCF(planeName)[0]
        p.changeDynamics(self.ground, -1, lateralFriction=1)
        return [self.ground]

class StadiumScene(Scene):
    """
    A simple stadium scene for debugging
    """
    def load(self):
        self.build_graph = False
        self.is_interactive = False
        filename = os.path.join(pybullet_data.getDataPath(), "stadium_no_collision.sdf")
        self.stadium = p.loadSDF(filename)
        planeName = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.ground = p.loadMJCF(planeName)[0]
        pos, orn = p.getBasePositionAndOrientation(self.ground)
        p.resetBasePositionAndOrientation(self.ground, [pos[0], pos[1], pos[2] - 0.005], orn)
        p.changeVisualShape(self.ground, -1, rgbaColor=[1, 1, 1, 0.5])
        return list(self.stadium) + [self.ground]

    def get_random_floor(self):
        return 0

    def get_random_point(self, random_height=False):
        return self.get_random_point_floor(0, random_height)

    def get_random_point_floor(self, floor, random_height=False):
        del floor
        return 0, np.array([
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5),
            np.random.uniform(0.4, 0.8) if random_height else 0.0
        ])

    def get_floor_height(self, floor):
        del floor
        return 0.0

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        logging.warning('WARNING: trying to compute the shortest path in StadiumScene (assuming empty space)')
        shortest_path = np.stack((source_world, target_world))
        geodesic_distance = l2_distance(source_world, target_world)
        return shortest_path, geodesic_distance

    def reset_floor(self, floor=0, additional_elevation=0.05, height=None):
        return



class BuildingScene(Scene):
    """
    Gibson Environment building scenes
    """
    def __init__(self,
                 model_id,
                 trav_map_resolution=0.1,
                 trav_map_erosion=2,
                 build_graph=True,
                 is_interactive=False,
                 num_waypoints=10,
                 waypoint_resolution=0.2,
                 pybullet_load_texture=False,
                 ):
        """
        Load a building scene and compute traversability

        :param model_id: Scene id
        :param trav_map_resolution: traversability map resolution
        :param trav_map_erosion: erosion radius of traversability areas, should be robot footprint radius
        :param build_graph: build connectivity graph
        :param is_interactive: whether the scene is interactive. If so, we will replace the annotated objects with the corresponding CAD models and add floor planes with the original floor texture.
        :param num_waypoints: number of way points returned
        :param waypoint_resolution: resolution of adjacent way points
        :param pybullet_load_texture: whether to load texture into pybullet. This is for debugging purpose only and does not affect what the robots see
        """
        logging.info("Building scene: {}".format(model_id))
        self.model_id = model_id
        self.trav_map_default_resolution = 0.01  # each pixel represents 0.01m
        self.trav_map_resolution = trav_map_resolution
        self.trav_map_original_size = None
        self.trav_map_size = None
        self.trav_map_erosion = trav_map_erosion
        self.build_graph = build_graph
        self.is_interactive = is_interactive
        self.num_waypoints = num_waypoints
        self.waypoint_interval = int(waypoint_resolution / trav_map_resolution)
        self.mesh_body_id = None
        self.floor_body_ids = []
        self.pybullet_load_texture = pybullet_load_texture

    def load_floor_metadata(self):
        """
        Load floor metadata
        """
        floor_height_path = os.path.join(get_model_path(self.model_id), 'floors.txt')
        if not os.path.isfile(floor_height_path):
            raise Exception('floors.txt cannot be found in model: {}'.format(self.model_id))
        with open(floor_height_path, 'r') as f:
            self.floors = sorted(list(map(float, f.readlines())))
            logging.debug('Floors {}'.format(self.floors))

    def load_scene_mesh(self):
        """
        Load scene mesh
        """
        if self.is_interactive:
            filename = os.path.join(get_model_path(self.model_id), "mesh_z_up_cleaned.obj")
        else:
            filename = os.path.join(get_model_path(self.model_id), "mesh_z_up_downsampled.obj")
            if not os.path.isfile(filename):
                filename = os.path.join(get_model_path(self.model_id), "mesh_z_up.obj")

        collision_id = p.createCollisionShape(p.GEOM_MESH,
                                              fileName=filename,
                                              flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        if self.pybullet_load_texture:
            visual_id = p.createVisualShape(p.GEOM_MESH,
                                            fileName=filename)
            texture_filename = get_texture_file(filename)
            if texture_filename is not None:
                texture_id = p.loadTexture(texture_filename)
            else:
                texture_id = -1
        else:
            visual_id = -1
            texture_id = -1

        self.mesh_body_id = p.createMultiBody(baseCollisionShapeIndex=collision_id,
                                              baseVisualShapeIndex=visual_id)
        p.changeDynamics(self.mesh_body_id, -1, lateralFriction=1)

        if self.pybullet_load_texture:
            if texture_id != -1:
                p.changeVisualShape(self.mesh_body_id,
                                -1,
                                textureUniqueId=texture_id)

    def load_floor_planes(self):
        if self.is_interactive:
            for f in range(len(self.floors)):
                # load the floor plane with the original floor texture for each floor
                plane_name = os.path.join(get_model_path(self.model_id), "plane_z_up_{}.obj".format(f))
                collision_id = p.createCollisionShape(p.GEOM_MESH,
                                                      fileName=plane_name)
                visual_id = p.createVisualShape(p.GEOM_MESH,
                                                fileName=plane_name)
                texture_filename = get_texture_file(plane_name)
                if texture_filename is not None:
                    texture_id = p.loadTexture(texture_filename)
                else:
                    texture_id = -1
                floor_body_id = p.createMultiBody(baseCollisionShapeIndex=collision_id,
                                                  baseVisualShapeIndex=visual_id)
                if texture_id != -1:
                    p.changeVisualShape(floor_body_id,
                                    -1,
                                    textureUniqueId=texture_id)
                floor_height = self.floors[f]
                p.resetBasePositionAndOrientation(floor_body_id,
                                                  posObj=[0, 0, floor_height],
                                                  ornObj=[0, 0, 0, 1])

                # Since both the floor plane and the scene mesh have mass 0 (static),
                # PyBullet seems to have already disabled collision between them.
                # Just to be safe, explicit disable collision between them.
                p.setCollisionFilterPair(self.mesh_body_id, floor_body_id, -1, -1, enableCollision=0)

                self.floor_body_ids.append(floor_body_id)
        else:
            # load the default floor plane (only once) and later reset it to different floor heiights
            plane_name = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
            floor_body_id = p.loadMJCF(plane_name)[0]
            p.resetBasePositionAndOrientation(floor_body_id,
                                              posObj=[0, 0, 0],
                                              ornObj=[0, 0, 0, 1])
            p.setCollisionFilterPair(self.mesh_body_id, floor_body_id, -1, -1, enableCollision=0)
            self.floor_body_ids.append(floor_body_id)

    def load_trav_map(self):
        self.floor_map = []
        self.floor_graph = []
        for f in range(len(self.floors)):
            trav_map = np.array(Image.open(
                os.path.join(get_model_path(self.model_id), 'floor_trav_{}.png'.format(f))
            ))
            obstacle_map = np.array(Image.open(
                os.path.join(get_model_path(self.model_id), 'floor_{}.png'.format(f))
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
                graph_file = os.path.join(get_model_path(self.model_id), 'floor_trav_{}.p'.format(f))
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
                                neighbors = [(i - 1, j - 1), (i, j - 1), (i + 1, j - 1), (i - 1, j)]
                                for n in neighbors:
                                    if 0 <= n[0] < self.trav_map_size and 0 <= n[1] < self.trav_map_size and trav_map[n[0], n[1]] > 0:
                                        g.add_edge(n, (i, j), weight=l2_distance(n, (i, j)))

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

    def load_scene_objects(self):
        if not self.is_interactive:
            return

        self.scene_objects = []
        self.scene_objects_pos = []
        scene_path = get_model_path(self.model_id)
        urdf_files = [item for item in os.listdir(scene_path) if item[-4:] == 'urdf' and item != 'scene.urdf']
        position_files = [item[:-4].replace('alignment_centered', 'pos') + 'txt' for item in urdf_files]
        for urdf_file, position_file in zip(urdf_files, position_files):
            logging.info('Loading urdf file {}'.format(urdf_file))
            with open(os.path.join(scene_path, position_file)) as f:
                pos = np.array([float(item) for item in f.readlines()[0].strip().split()])
                obj = InteractiveObj(os.path.join(scene_path, urdf_file))
                obj.load()
                self.scene_objects.append(obj)
                self.scene_objects_pos.append(pos)

    def load_scene_urdf(self):
        self.mesh_body_id = p.loadURDF(os.path.join(get_model_path(self.model_id), 'scene.urdf'))

    def has_scene_urdf(self):
        return os.path.exists(os.path.join(get_model_path(self.model_id), 'scene.urdf'))

    def load(self):
        """
        Initialize scene
        """
        self.load_floor_metadata()
        if self.has_scene_urdf():
            self.load_scene_urdf()
        else:
            self.load_scene_mesh()
            self.load_floor_planes()

        self.load_trav_map()
        self.load_scene_objects()
        self.reset_scene_objects()

        return [self.mesh_body_id] + self.floor_body_ids

    def get_random_floor(self):
        return np.random.randint(0, high=len(self.floors))

    def get_random_point(self, random_height=False):
        floor = self.get_random_floor()
        return self.get_random_point_floor(floor, random_height)

    def get_random_point_floor(self, floor, random_height=False):
        trav = self.floor_map[floor]
        trav_space = np.where(trav == 255)
        idx = np.random.randint(0, high=trav_space[0].shape[0])
        xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
        x, y = self.map_to_world(xy_map)
        z = self.floors[floor]
        if random_height:
            z += np.random.uniform(0.4, 0.8)
        return floor, np.array([x, y, z])

    def map_to_world(self, xy):
        axis = 0 if len(xy.shape) == 1 else 1
        return np.flip((xy - self.trav_map_size / 2.0) * self.trav_map_resolution, axis=axis)

    def world_to_map(self, xy):
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
            closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - target_map, axis=1))])
            g.add_edge(closest_node, target_map, weight=l2_distance(closest_node, target_map))

        if not g.has_node(source_map):
            nodes = np.array(g.nodes)
            closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - source_map, axis=1))])
            g.add_edge(closest_node, source_map, weight=l2_distance(closest_node, source_map))

        path_map = np.array(nx.astar_path(g, source_map, target_map, heuristic=l2_distance))

        path_world = self.map_to_world(path_map)
        geodesic_distance = np.sum(np.linalg.norm(path_world[1:] - path_world[:-1], axis=1))
        path_world = path_world[::self.waypoint_interval]

        if not entire_path:
            path_world = path_world[:self.num_waypoints]
            num_remaining_waypoints = self.num_waypoints - path_world.shape[0]
            if num_remaining_waypoints > 0:
                remaining_waypoints = np.tile(target_world, (num_remaining_waypoints, 1))
                path_world = np.concatenate((path_world, remaining_waypoints), axis=0)

        return path_world, geodesic_distance

    def reset_floor(self, floor=0, additional_elevation=0.02, height=None):
        if self.is_interactive:
            # loads the floor plane with the original floor texture for each floor, no need to reset_floor
            return

        height = height if height is not None else self.floors[floor] + additional_elevation
        p.resetBasePositionAndOrientation(self.floor_body_ids[0],
                                          posObj=[0, 0, height],
                                          ornObj=[0, 0, 0, 1])

    def reset_scene_objects(self):
        if not self.is_interactive:
            # does not have objects in the scene
            return

        for obj, pos in zip(self.scene_objects, self.scene_objects_pos):
            obj.set_position_orientation(pos,
                                         [0, 0, 0, 1])

    def get_floor_height(self, floor):
        return self.floors[floor]

