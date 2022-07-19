import logging
import os

import numpy as np
import pybullet as p
import pybullet_data

from igibson.scenes.scene_base import Scene
from igibson.utils.constants import SemanticClass
from igibson.utils.utils import l2_distance

log = logging.getLogger(__name__)


class EmptyScene(Scene):
    """
    An empty scene for debugging.
    """

    def __init__(self, render_floor_plane=True, floor_plane_rgba=[1.0, 1.0, 1.0, 1.0]):
        super(EmptyScene, self).__init__()
        self.objects = []
        self.render_floor_plane = render_floor_plane
        self.floor_plane_rgba = floor_plane_rgba

    def get_objects(self):
        return list(self.objects)

    def _add_object(self, obj):
        self.objects.append(obj)

    def _load(self, simulator):
        plane_file = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.floor_body_ids += [p.loadMJCF(plane_file)[0]]
        p.changeDynamics(self.floor_body_ids[0], -1, lateralFriction=1)
        # White floor plane for visualization purpose if needed.
        p.changeVisualShape(self.floor_body_ids[0], -1, rgbaColor=self.floor_plane_rgba)

        if self.render_floor_plane:
            for id in self.floor_body_ids:
                simulator.load_object_in_renderer(
                    None, id, SemanticClass.SCENE_OBJS, use_pbr=False, use_pbr_mapping=False
                )

        # Load additional objects & merge body IDs.
        additional_object_body_ids = [x for obj in self.objects for x in obj.load(simulator)]
        return self.floor_body_ids + additional_object_body_ids

    def get_random_point(self, floor=None):
        """
        Get a random point in the region of [-5, 5] x [-5, 5].
        """
        return floor, np.array(
            [
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                0.0,
            ]
        )

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        """
        Get a trivial shortest path because the scene is empty.
        """
        log.debug("WARNING: trying to compute the shortest path in EmptyScene (assuming empty space)")
        shortest_path = np.stack((source_world, target_world))
        geodesic_distance = l2_distance(source_world, target_world)
        return shortest_path, geodesic_distance
