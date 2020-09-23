import pybullet_data
import pybullet as p
import os
from gibson2.scenes.scene_base import Scene
import logging
import numpy as np
from gibson2.utils.utils import l2_distance


class EmptyScene(Scene):
    """
    A empty scene for debugging
    """

    def __init__(self):
        super().__init__()

    def load(self):
        plane_file = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.floor_body_ids += [p.loadMJCF(plane_file)[0]]
        p.changeDynamics(self.floor_body_ids[0], -1, lateralFriction=1)
        # white floor plane for visualization purpose if needed
        p.changeVisualShape(self.floor_body_ids[0], -1, rgbaColor=[1, 1, 1, 1])
        return self.floor_body_ids

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        logging.warning('WARNING: trying to compute the shortest path in EmptyScene (assuming empty space)')
        shortest_path = np.stack((source_world, target_world))
        geodesic_distance = l2_distance(source_world, target_world)
        return shortest_path, geodesic_distance
