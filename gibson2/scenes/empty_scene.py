import pybullet_data
import pybullet as p
import os
from gibson2.scenes.scene_base import Scene


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
        # white floor plane for visualization purpose if needed
        p.changeVisualShape(self.ground, -1, rgbaColor=[1, 1, 1, 1])
        return [self.ground]