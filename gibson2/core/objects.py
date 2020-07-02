import pybullet as p
import os
import gibson2
import numpy as np


class BoxShape(object):
    def __init__(self, pos=[1, 2, 3], dim=[1, 2, 3], rgba_color=[1.0, 1.0, 1.0, 1.0], mass=1000):
        self.basePos = pos
        self.dimension = dim

        self.rgba_color = rgba_color
        self.collision_id = None

        self.body_id = None

        self.mass = mass


    def load(self):
        baseOrientation = [0, 0, 0, 1]

        self.collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.dimension)
        visualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=self.dimension, rgbaColor=self.rgba_color)

        self.body_id = p.createMultiBody(baseMass=self.mass,
            baseCollisionShapeIndex=self.collision_id,
            baseVisualShapeIndex=visualShapeId,
            basePosition=self.basePos,
            baseOrientation=baseOrientation)

        p.resetBasePositionAndOrientation(self.body_id, self.basePos, baseOrientation)
        return self.body_id

    def set_color(self, color):
        p.changeVisualShape(self.body_id, -1, rgbaColor=color)


    def set_position(self, pos):
        _, org_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, org_orn)


    def get_position(self):
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        return pos

    def get_dimensions(self):
        return self.dimension



class VisualObject(object):
    def __init__(self,
                 visual_shape=p.GEOM_SPHERE,
                 rgba_color=[1, 0, 0, 0.5],
                 radius=1.0,
                 half_extents=[1, 1, 1],
                 length=1,
                 initial_offset=[0, 0, 0]):
        self.visual_shape = visual_shape
        self.rgba_color = rgba_color
        self.radius = radius
        self.half_extents = half_extents
        self.length = length
        self.initial_offset = initial_offset

    def load(self):
        if self.visual_shape == p.GEOM_BOX:
            shape = p.createVisualShape(self.visual_shape,
                                        rgbaColor=self.rgba_color,
                                        halfExtents=self.half_extents,
                                        visualFramePosition=self.initial_offset)
        elif self.visual_shape in [p.GEOM_CYLINDER, p.GEOM_CAPSULE]:
            shape = p.createVisualShape(self.visual_shape,
                                        rgbaColor=self.rgba_color,
                                        radius=self.radius,
                                        length=self.length,
                                        visualFramePosition=self.initial_offset)
        else:
            shape = p.createVisualShape(self.visual_shape,
                                        rgbaColor=self.rgba_color,
                                        radius=self.radius,
                                        visualFramePosition=self.initial_offset)
        self.body_id = p.createMultiBody(baseVisualShapeIndex=shape, baseCollisionShapeIndex=-1)

        return self.body_id

    def set_position(self, pos, new_orn=None):
        if new_orn is None:
            _, new_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, new_orn)

    def set_color(self, color):
        p.changeVisualShape(self.body_id, -1, rgbaColor=color)
