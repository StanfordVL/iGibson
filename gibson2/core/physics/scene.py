import pybullet as p
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet_data
from gibson2.utils.assets_utils import get_model_path, get_texture_file, get_ig_scene_path
from gibson2.utils.utils import l2_distance, get_transform_from_xyz_rpy, quatXYZWFromRotMat, get_rpy_from_transform
from gibson2.core.physics.interactive_objects import InteractiveObj, URDFObject

import numpy as np
from PIL import Image
import cv2
import networkx as nx
import pickle
import logging
import xml.etree.ElementTree as ET

import gibson2
import math

class Scene:
    def __init__(self):
        self.is_interactive = False

    def load(self):
        raise NotImplementedError()

class EmptyScene(Scene):
    """
    A empty scene for debugging
    """
    def __init__(self):
        super().__init__()
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
    def __init__(self):
        super().__init__()

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
        super().__init__()
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

def parse_urdf(tree):
    parent_map = {} # map from name of child to name of its parent, joint name and type of connection
    child_map = {} # map from name of parent to list of names of children, joint names and types of connection
    joint_map = {} # map from name of joint to names of parent and child and type
    single_link = []

    single_link_urdf = True
    for joint in tree.iter("joint"): # We iterate over joints to build maps
        single_link_urdf = False
        parent_name = joint.find("parent").attrib["link"]
        child_name = joint.find("child").attrib["link"]
        joint_name = joint.attrib["name"]
        joint_type = joint.attrib["type"]
        
        parent_map[child_name] = (parent_name, joint_name, joint_type)
        if parent_name in child_map:
            child_map[parent_name].append((child_name, joint_name, joint_type))
        else:
            child_map[parent_name] = [(child_name, joint_name, joint_type)]

        joint_xyz = np.array([float(val) for val in joint.find("origin").attrib["xyz"].split(" ")])

        if 'rpy' in joint.find("origin").attrib:
            joint_rpy = np.array([float(val) for val in joint.find("origin").attrib["rpy"].split(" ")])
        else:
            joint_rpy = np.array([0.,0.,0.])

        joint_frame = get_transform_from_xyz_rpy(joint_xyz, joint_rpy)
        joint_map[joint_name] = (parent_name, child_name, joint_type, joint_frame)

    if single_link_urdf:
        single_link = [tree.find("link").attrib["name"]]

    return (parent_map, child_map, joint_map, single_link)

def splitter(parent_map, child_map, joint_map, single_child_link):
    new_single_child_link = []
    for (joint_name, joint_tuple) in joint_map.items():
        logging.debug("Joint: ", joint_name)
        if joint_tuple[2] == "floating":

            logging.debug("Splitting floating joint")
            # separate into the two parts and call recursively splitter with each part
            parent_of_floating = joint_tuple[0]
            child_of_floating = joint_tuple[1]

            # If the children of float is not parent of any link, we add it to the sengle_child_link
            if child_of_floating not in child_map.keys():
                new_single_child_link += [child_of_floating]
            
            parent_map1 = {}
            child_map1 = {}
            joint_map1 = {}
            parent_map2 = {}
            child_map2 = {}
            joint_map2 = {}
            
            # Find all links "down" the floating joint
            logging.debug("Finding children")
            logging.info("Child of floating: " + child_of_floating )
            all_children = [child_of_floating]
            children_rec = [child_of_floating]
            while len(children_rec) != 0:
                new_children_rec = []
                for child in children_rec:
                    if child in child_map:
                        new_children_rec += child_map[child]
                        
                all_children += [new_child[0] for new_child in new_children_rec]
                children_rec = [new_child[0] for new_child in new_children_rec]

            logging.info("All children of the floating joint: " + " ".join(all_children))

            # Separate joints in map1 and map2
            # The ones in map2 are the ones with the child pointing to one of the links "down" the floating joint
            logging.debug("Splitting joints")
            for (joint_name2, joint_tuple2) in joint_map.items():
                if joint_name2 != joint_name:
                    if joint_tuple2[1] in all_children:
                        joint_map2[joint_name2] = joint_tuple2
                    else:
                        joint_map1[joint_name2] = joint_tuple2                 

            # Separate children into map1 and map2
            # Careful with the child_map because every key of the dict (name of parent) points to a list of children
            logging.debug("Splitting children")
            for parent in child_map: # iterate all links that are parent of 1 or more joints
                child_list = child_map[parent]  # for each parent, get the list of children
                if parent in all_children:  # if the parent link was in the list of all children of the floating joint
                    child_map2[parent] = child_list # save the list as list of children of the parent link in the children floating suburdf
                else: #otherwise, it is one of the links parents of the floating joint
                    child_map1[parent] = [item for item in child_list if item[0] != child_of_floating] # save the list as the list of
                    # children of the parent in the parent floating suburdf, except the children connected by the floating joint

            # Separate parents into map1 and map2
            for child in parent_map:
                if child != child_of_floating:
                    if child in all_children:
                        parent_map2[child] = parent_map[child]
                    else:
                        parent_map1[child] = parent_map[child]              

            ret1 = splitter(parent_map1, child_map1, joint_map1, [])
            ret2 = splitter(parent_map2, child_map2, joint_map2, new_single_child_link)
            ret = ret1 + ret2
            return ret
    return [(parent_map, child_map, joint_map, single_child_link)]

def round_up(n, decimals=0): 
    multiplier = 10 ** decimals 
    return math.ceil(n * multiplier) / multiplier

def transform_element_xyzrpy(element, transformation):
    element_xyz = np.array([float(val) for val in element.find("origin").attrib["xyz"].split(" ")])
    if 'rpy' in element.find("origin").attrib:
        element_rpy = np.array([float(val) for val in element.find("origin").attrib["rpy"].split(" ")])
    else:
        element_rpy = np.array([0.,0.,0.])
    element_transform = get_transform_from_xyz_rpy(element_xyz, element_rpy)
    total_transform = np.dot(transformation, element_transform)
    element.find("origin").attrib["xyz"] = "{0:f} {1:f} {2:f}".format(*total_transform[0:3,3])
    transform_rpy = get_rpy_from_transform(total_transform)
    element.find("origin").attrib["rpy"] = "{0:f} {1:f} {2:f}".format(*transform_rpy)

def merge_fixed_joints(tree):
    while True:
        fixed_joints = [joint for joint in tree.findall("joint") if joint.attrib["type"] == "fixed"]
        if len(fixed_joints) == 0:
            break
        else:
            fixed_joint = fixed_joints[0]

            joint_xyz = np.array([float(val) for val in fixed_joint.find("origin").attrib["xyz"].split(" ")])

            if 'rpy' in fixed_joint.find("origin").attrib:
                joint_rpy = np.array([float(val) for val in fixed_joint.find("origin").attrib["rpy"].split(" ")])
            else:
                joint_rpy = np.array([0.,0.,0.])

            joint_frame = get_transform_from_xyz_rpy(joint_xyz, joint_rpy)

            # The transformation needs to be applied to all positional elements defined with respect to the fixed_joint frame
            # This includes: 
            #   - link elements (visual and geom, we ignore inertia)
            #   - next joint elements
            child_link_name = fixed_joint.find("child").attrib["link"]
            child_link = [link for link in tree.findall("link") if link.attrib["name"] == child_link_name][0]
            parent_link_name = fixed_joint.find("parent").attrib["link"]
            parent_link = [link for link in tree.findall("link") if link.attrib["name"] == parent_link_name][0]
            for visual_elem in child_link.iter("visual"):
                transform_element_xyzrpy(visual_elem, joint_frame)
                parent_link.append(visual_elem)                

            for collision_elem in child_link.iter("collision"):
                transform_element_xyzrpy(collision_elem, joint_frame)
                parent_link.append(collision_elem)

            for joint2 in tree.iter("joint"): 
                parent_name = joint2.find("parent").attrib["link"]
                # Search for a joint where the child of the "fixed" joint is the parent
                if parent_name == child_link_name: 
                    transform_element_xyzrpy(joint2, joint_frame)
                    joint2.find("parent").attrib["link"] = parent_link_name

            tree.getroot().remove(child_link)
            tree.getroot().remove(fixed_joint)

def save_urdfs_without_floating_joints(tree, file_prefix, merge_fj):

    if merge_fj:
        merge_fixed_joints(tree)

    #Pybullet doesn't read floating joints
    #Find them and separate into different objects
    (parent_map, child_map, joint_map, single_floating_links) = parse_urdf(tree)

    # Call recursively to split the tree into connected parts without floating joints
    splitted_maps = splitter(parent_map, child_map, joint_map, single_floating_links)

    extended_splitted_dict = {}
    world_idx = 0
    for (count, split) in enumerate(splitted_maps):
        all_links = []
        for parent in split[0]:
            if parent not in all_links:
                all_links.append(parent)
        for child in split[1]:
            if child not in all_links:
                all_links.append(child)
        for link in split[3]:
            if link not in all_links:
                all_links.append(link)
        extended_splitted_dict[count] = ((split[0], split[1], split[2], all_links, np.eye(4)))
        if "world" in all_links:
            world_idx = count
            logging.debug("World idx: ", world_idx)

    # Find the transformations, starting from "world" link
    for (joint_name, joint_tuple) in joint_map.items():
        logging.debug("Joint: " + joint_name)
        if joint_tuple[2] == "floating":
            logging.debug("floating")
            parent_name = joint_tuple[0]
            transformation = joint_tuple[3]

            while parent_name in parent_map.keys(): #!= "world": #When the parent_name link is not child of any other joint, we stop
                # Find the joint where the link with name "parent_name" is child
                joint_up = [joint for joint in tree.findall("joint") if joint.find("child").attrib["link"] == parent_name][0]
                joint_transform = joint_map[joint_up.attrib["name"]][3]
                transformation = np.dot(joint_transform, transformation)
                parent_name = joint_map[joint_up.attrib["name"]][0]

            child_name = joint_tuple[1]
            for esd in extended_splitted_dict:
                if child_name in extended_splitted_dict[esd][3]:
                    extended_splitted_dict[esd] = (extended_splitted_dict[esd][0], extended_splitted_dict[esd][1], extended_splitted_dict[esd][2],
                        extended_splitted_dict[esd][3], transformation)

    logging.info("Number of splits: " + str(len(extended_splitted_dict)))
    logging.info("Instantiating scene into the following urdfs:")
    urdfs_no_floating = {}
    for esd_key in extended_splitted_dict:            
        xml_tree_parent = ET.ElementTree(ET.fromstring('<robot name="split_' + str(esd_key) + '"></robot>'))
        logging.info("links " + " ".join(extended_splitted_dict[esd_key][3]))        

        for link_name in extended_splitted_dict[esd_key][3]:
            link_to_add = [link for link in tree.findall("link") if link.attrib["name"] == link_name][0]
            xml_tree_parent.getroot().append(link_to_add)

        for joint_name in extended_splitted_dict[esd_key][2]:
            joint_to_add = [joint for joint in tree.findall("joint") if joint.attrib["name"] == joint_name][0]
            xml_tree_parent.getroot().append(joint_to_add)

        # Copy the elements that are not joint or link (e.g. material)
        for item in list(tree.getroot()):
            if item.tag not in ['link', 'joint']:                    
                xml_tree_parent.getroot().append(item)
        
        urdf_file_name = file_prefix + "_" + str(esd_key)+ ".urdf"
        urdfs_no_floating[esd_key] = (urdf_file_name, extended_splitted_dict[esd_key][4]) # Change 0 by the pose of this branch
        xml_tree_parent.write(urdf_file_name)
        logging.info(urdf_file_name)
    return urdfs_no_floating


class iGSDFScene(Scene):
    """
    Create a scene defined with iGibson Scene Description Format (igsdf).

    iGSDF is an extension of URDF that we use to define an interactive scene. It has support for URDF scaling,
    URDF nesting and randomization.

    """
    def __init__(self, scene_name):
        super().__init__()
        self.scene_file = get_ig_scene_path(scene_name) + "/" + scene_name + ".urdf"
        self.scene_tree = ET.parse(self.scene_file)
        self.links = []
        self.joints = []
        self.links_by_name = {}
        self.joints_by_name = {}
        self.nested_urdfs = []

        # We have two ways of processing:
        # 1) merge all urdfs into a composed urdf, then separate the floating parts
        # 2) have one urdf per object
        self.compose_urdf = True 

        # If this flag is true, we merge fixed joints into unique bodies
        self.merge_fj = True

        self.random_groups = {}

        # Parse all the special link entries in the root URDF that defines the scene
        for link in self.scene_tree.findall('link'):

            if 'category' in link.attrib:
                embedded_urdf = URDFObject(link, self.random_groups)

                base_link_name = link.attrib['name']

                for link_emb in embedded_urdf.object_tree.iter('link'):
                    if link_emb.attrib['name'] == "base_link": 
                        # The base_link get renamed as the link tag indicates
                        if self.compose_urdf:
                            # First extending the link tag in the fused urdf, then renaming back
                            link.attrib.update(link_emb.attrib)
                            link.attrib['name'] = base_link_name
                            link.extend(list(link_emb))
                        else:
                            # Just change the name of the base link in the embedded urdf
                            link_emb.attrib['name'] = base_link_name
                    else:
                        # The other links get also renamed to add the name of the link tag as prefix
                        # This allows us to load several instances of the same object
                        link_emb.attrib['name'] = base_link_name + "_" + link_emb.attrib['name']
                        if self.compose_urdf:
                            self.scene_tree.getroot().append(link_emb)

                for joint_emb in embedded_urdf.object_tree.iter('joint'):
                    # We change the joint name
                    joint_emb.attrib["name"] = base_link_name + "_" + joint_emb.attrib["name"]
                    # We change the child link names
                    for child_emb in joint_emb.findall('child'):
                        if child_emb.attrib['link'] == "base_link":
                            child_emb.attrib['link'] = base_link_name
                        else:
                            child_emb.attrib['link'] = base_link_name + "_" + child_emb.attrib['link']
                    # and the parent link names
                    for parent_emb in joint_emb.findall('parent'):
                        if parent_emb.attrib['link'] == "base_link":
                            parent_emb.attrib['link'] = base_link_name
                        else:
                            parent_emb.attrib['link'] = base_link_name + "_" + parent_emb.attrib['link']

                    # and add the joint
                    if self.compose_urdf:
                        self.scene_tree.getroot().append(joint_emb)

                if self.compose_urdf:
                    for item in list(embedded_urdf.object_tree.getroot()):
                        if item.tag not in ['link', 'joint']:                     
                            self.scene_tree.getroot().append(item)
                else:
                    # Deal with the joint connecting the embedded urdf to the main link (world or building)
                    urdf_file_name_prefix = gibson2.ig_dataset_path + "/scene_instances/" + base_link_name # + ".urdf"

                    # Find the joint in the main urdf that defines the connection to the embedded urdf
                    for joint in self.scene_tree.iter('joint'):
                        if joint.find('child').attrib['link'] == base_link_name:
                            joint_frame = np.eye(4)

                            # if the joint is not floating, we add the joint and a link to the embedded urdf
                            if joint.attrib['type'] != "floating": 
                                embedded_urdf.object_tree.getroot().append(joint)
                                parent_link = ET.SubElement(embedded_urdf.object_tree.getroot(), "link", 
                                    dict([("name", joint.find('parent').attrib['link'])]))#"world")]))   

                            # if the joint is floating, we save the transformation in the floating joint to be used when we load the 
                            # embedded urdf                            
                            else:
                                joint_xyz = np.array([float(val) for val in joint.find("origin").attrib["xyz"].split(" ")])

                                if 'rpy' in joint.find("origin").attrib:
                                    joint_rpy = np.array([float(val) for val in joint.find("origin").attrib["rpy"].split(" ")])
                                else:
                                    joint_rpy = np.array([0.,0.,0.])
                                joint_frame = get_transform_from_xyz_rpy(joint_xyz, joint_rpy)

                            # Deal with floating joints inside the embedded urdf
                            urdfs_no_floating = save_urdfs_without_floating_joints(embedded_urdf.object_tree, 
                                gibson2.ig_dataset_path + "/scene_instances/" + base_link_name, self.merge_fj)  

                            # append a new tuple of file name of the instantiated embedded urdf 
                            # and the transformation (!= None if its connection was floating)
                            for urdf in urdfs_no_floating:
                                transformation = np.dot(joint_frame, urdfs_no_floating[urdf][1])
                                self.nested_urdfs += [(urdfs_no_floating[urdf][0], transformation)]

        if self.compose_urdf:
            self.file_ctr = 0            
            urdf_file_name = gibson2.ig_dataset_path + "/scene_instances/scene_instance_full.urdf"
            self.scene_tree.write(urdf_file_name)
            self.urdfs_no_floating = save_urdfs_without_floating_joints(self.scene_tree, 
                gibson2.ig_dataset_path + "/scene_instances/scene_instance", self.merge_fj)  


    def load(self):
        body_ids = []
        if self.compose_urdf:
            for urdf in self.urdfs_no_floating:
                logging.info("Loading " + self.urdfs_no_floating[urdf][0])
                body_id = p.loadURDF(self.urdfs_no_floating[urdf][0]) 
                                 #flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
                logging.info("Moving URDF to " + np.array_str(self.urdfs_no_floating[urdf][1]))
                transformation = self.urdfs_no_floating[urdf][1]
                oriii = np.array(quatXYZWFromRotMat(transformation[0:3,0:3]))
                transl = transformation[0:3,3]
                p.resetBasePositionAndOrientation(body_id, transl, oriii)
                self.mass = p.getDynamicsInfo(body_id, -1)[0]
                activationState = p.ACTIVATION_STATE_ENABLE_SLEEPING+p.ACTIVATION_STATE_SLEEP
                p.changeDynamics(body_id, -1, activationState=activationState)
                body_ids += [body_id]
        else:
            for urdf in self.nested_urdfs:
                logging.info("Loading " + urdf[0])
                body_id = p.loadURDF(urdf[0]) 
                                 #flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
                logging.info("Moving URDF to " + np.array_str(urdf[1]))
                transformation = urdf[1]
                oriii = np.array(quatXYZWFromRotMat(transformation[0:3,0:3]))
                transl = transformation[0:3,3]
                p.resetBasePositionAndOrientation(body_id, transl, oriii)

                self.mass = p.getDynamicsInfo(body_id, -1)[0]
                activationState = p.ACTIVATION_STATE_ENABLE_SLEEPING+p.ACTIVATION_STATE_SLEEP
                p.changeDynamics(body_id, -1, activationState=activationState)
                body_ids += [body_id]
        return body_ids