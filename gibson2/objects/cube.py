import math

from gibson2.objects.object_base import Object
import pybullet as p


class Cube(Object):
    def __init__(self, pos=[1, 2, 3], dim=[1, 2, 3], visual_only=False, mass=1000, color=[1, 1, 1, 1]):
        super(Cube, self).__init__()
        self.basePos = pos
        self.dimension = dim
        self.visual_only = visual_only
        self.mass = mass
        self.color = color

    def _load(self):
        baseOrientation = [0, 0, 0, 1]
        colBoxId = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=self.dimension)
        visualShapeId = p.createVisualShape(
            p.GEOM_BOX, halfExtents=self.dimension, rgbaColor=self.color)
        if self.visual_only:
            body_id = p.createMultiBody(baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=visualShapeId)
        else:
            body_id = p.createMultiBody(baseMass=self.mass,
                                        baseCollisionShapeIndex=colBoxId,
                                        baseVisualShapeIndex=visualShapeId)

        p.resetBasePositionAndOrientation(
            body_id, self.basePos, baseOrientation)

        return body_id


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier