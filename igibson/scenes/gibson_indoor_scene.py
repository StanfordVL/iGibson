import logging
import pickle
import networkx as nx
import cv2
from PIL import Image
import numpy as np
from igibson.objects.articulated_object import ArticulatedObject, URDFObject
from igibson.utils.utils import l2_distance, get_transform_from_xyz_rpy, quatXYZWFromRotMat
from igibson.utils.assets_utils import get_scene_path, get_texture_file, get_ig_scene_path
import pybullet_data
import pybullet as p
import os
from igibson.scenes.indoor_scene import IndoorScene


class StaticIndoorScene(IndoorScene):
    """
    Static indoor scene class for iGibson.
    Contains the functionalities for navigation such as shortest path computation
    """

    def __init__(self,
                 scene_id,
                 trav_map_resolution=0.1,
                 trav_map_erosion=2,
                 trav_map_type='with_obj',
                 build_graph=True,
                 num_waypoints=10,
                 waypoint_resolution=0.2,
                 pybullet_load_texture=False,
                 ):
        """
        Load a building scene and compute traversability

        :param scene_id: Scene id
        :param trav_map_resolution: traversability map resolution
        :param trav_map_erosion: erosion radius of traversability areas, should be robot footprint radius
        :param trav_map_type: type of traversability map, with_obj | no_obj
        :param build_graph: build connectivity graph
        :param num_waypoints: number of way points returned
        :param waypoint_resolution: resolution of adjacent way points
        :param pybullet_load_texture: whether to load texture into pybullet. This is for debugging purpose only and does not affect robot's observations
        """
        super(StaticIndoorScene, self).__init__(
            scene_id,
            trav_map_resolution,
            trav_map_erosion,
            trav_map_type,
            build_graph,
            num_waypoints,
            waypoint_resolution,
            pybullet_load_texture,
        )
        logging.info("StaticIndoorScene scene: {}".format(scene_id))

    def load_floor_metadata(self):
        """
        Load floor metadata
        """
        floor_height_path = os.path.join(
            get_scene_path(self.scene_id), 'floors.txt')
        if not os.path.isfile(floor_height_path):
            raise Exception(
                'floor_heights.txt cannot be found in model: {}'.format(self.scene_id))
        with open(floor_height_path, 'r') as f:
            self.floor_heights = sorted(list(map(float, f.readlines())))
            logging.debug('Floors {}'.format(self.floor_heights))

    def load_scene_mesh(self):
        """
        Load scene mesh
        """
        filename = os.path.join(get_scene_path(
            self.scene_id), "mesh_z_up_downsampled.obj")
        if not os.path.isfile(filename):
            filename = os.path.join(get_scene_path(
                self.scene_id), "mesh_z_up.obj")

        collision_id = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=filename,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        if self.pybullet_load_texture:
            visual_id = p.createVisualShape(
                p.GEOM_MESH,
                fileName=filename)
        else:
            visual_id = -1

        self.mesh_body_id = p.createMultiBody(
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id)
        p.changeDynamics(self.mesh_body_id, -1, lateralFriction=1)

        if self.pybullet_load_texture:
            texture_filename = get_texture_file(filename)
            if texture_filename is not None:
                texture_id = p.loadTexture(texture_filename)
                p.changeVisualShape(
                    self.mesh_body_id,
                    -1,
                    textureUniqueId=texture_id)

    def load_floor_planes(self):
        """
        Load additional floor planes (because the scene mesh can have bumpy floor surfaces)
        """
        # load the default floor plane (only once) and later reset it to different floor heiights
        plane_name = os.path.join(
            pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        floor_body_id = p.loadMJCF(plane_name)[0]
        p.resetBasePositionAndOrientation(floor_body_id,
                                          posObj=[0, 0, 0],
                                          ornObj=[0, 0, 0, 1])
        p.setCollisionFilterPair(
            self.mesh_body_id, floor_body_id, -1, -1, enableCollision=0)
        self.floor_body_ids.append(floor_body_id)

    def load(self):
        """
        Load the scene (including scene mesh and floor plane) into pybullet
        """
        self.load_floor_metadata()
        self.load_scene_mesh()
        self.load_floor_planes()

        self.load_trav_map(get_scene_path(self.scene_id))
        return [self.mesh_body_id] + self.floor_body_ids

    def get_random_floor(self):
        """
        Get a random floor

        :return: random floor number
        """
        return np.random.randint(0, high=len(self.floor_heights))

    def reset_floor(self, floor=0, additional_elevation=0.02, height=None):
        """
        Resets the floor plane to a new floor

        :param floor: Integer identifying the floor to move the floor plane to
        :param additional_elevation: Additional elevation with respect to the height of the floor
        :param height: Alternative parameter to control directly the height of the ground plane
        """
        height = height if height is not None \
            else self.floor_heights[floor] + additional_elevation
        p.resetBasePositionAndOrientation(self.floor_body_ids[0],
                                          posObj=[0, 0, height],
                                          ornObj=[0, 0, 0, 1])

    def get_floor_height(self, floor=0):
        """
        Return the current floor height (in meter)

        :return: current floor height
        """
        return self.floor_heights[floor]
