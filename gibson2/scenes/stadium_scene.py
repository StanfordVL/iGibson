import logging
import numpy as np
from gibson2.utils.utils import l2_distance
import pybullet_data
import pybullet as p
import os
from gibson2.scenes.scene_base import Scene


class StadiumScene(Scene):
    """
    A simple stadium scene for debugging
    """

    def __init__(self):
        super().__init__()

    def load(self):
        self.build_graph = False
        self.is_interactive = False
        filename = os.path.join(
            pybullet_data.getDataPath(), "stadium_no_collision.sdf")
        self.stadium = p.loadSDF(filename)
        planeName = os.path.join(
            pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.ground = p.loadMJCF(planeName)[0]
        pos, orn = p.getBasePositionAndOrientation(self.ground)
        p.resetBasePositionAndOrientation(
            self.ground, [pos[0], pos[1], pos[2] - 0.005], orn)
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
        logging.warning(
            'WARNING: trying to compute the shortest path in StadiumScene (assuming empty space)')
        shortest_path = np.stack((source_world, target_world))
        geodesic_distance = l2_distance(source_world, target_world)
        return shortest_path, geodesic_distance

    def reset_floor(self, floor=0, additional_elevation=0.05, height=None):
        return
