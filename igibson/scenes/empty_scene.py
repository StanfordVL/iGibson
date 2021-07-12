import pybullet_data
import pybullet as p
import os
from igibson.scenes.scene_base import Scene
import logging
import numpy as np
from igibson.utils.utils import l2_distance


class EmptyScene(Scene):
    """
    A empty scene for debugging
    """

    def __init__(self):
        super(EmptyScene, self).__init__()

    def load(self):
        """
        Load the scene into pybullet
        """
        plane_file = os.path.join(
            pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.floor_body_ids += [p.loadMJCF(plane_file)[0]]
        p.changeDynamics(self.floor_body_ids[0], -1, lateralFriction=1)
        # white floor plane for visualization purpose if needed
        p.changeVisualShape(self.floor_body_ids[0], -1, rgbaColor=[1, 1, 1, 1])
        return self.floor_body_ids

    def get_random_point(self, floor=None):
        """
        Get a random point in the region of [-5, 5] x [-5, 5]
        """
        return floor, np.array([
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5),
            0.0,
        ])

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        """
        Get a trivial shortest path because the scene is empty
        """
        logging.warning(
            'WARNING: trying to compute the shortest path in EmptyScene (assuming empty space)')
        shortest_path = np.stack((source_world, target_world))
        geodesic_distance = l2_distance(source_world, target_world)
        return shortest_path, geodesic_distance
