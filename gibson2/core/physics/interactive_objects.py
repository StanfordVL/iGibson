import pybullet as p
import os
import gibson2
import numpy as np


class YCBObject(object):
    def __init__(self, name, scale=1):
        self.filename = os.path.join(gibson2.assets_path, 'models', 'ycb', name,
                                     'textured_simple.obj')
        self.scale = scale

    def load(self):
        collision_id = p.createCollisionShape(p.GEOM_MESH,
                                              fileName=self.filename,
                                              meshScale=self.scale)
        body_id = p.createMultiBody(basePosition=[0, 0, 0],
                                    baseMass=0.1,
                                    baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=-1)
        return body_id


class ShapeNetObject(object):
    def __init__(self, path, scale=1., position=[0, 0, 0], orientation=[0, 0, 0]):
        self.filename = path

        self.scale = scale
        self.position = position
        self.orientation = orientation

        self._default_mass = 3.
        self._default_transform = {
            'position': [0, 0, 0],
            'orientation_quat': [1. / np.sqrt(2), 0, 0, 1. / np.sqrt(2)],
        }
        pose = p.multiplyTransforms(positionA=self.position,
                                    orientationA=p.getQuaternionFromEuler(self.orientation),
                                    positionB=self._default_transform['position'],
                                    orientationB=self._default_transform['orientation_quat'])
        self.pose = {
            'position': pose[0],
            'orientation_quat': pose[1],
        }

    def load(self):
        collision_id = p.createCollisionShape(p.GEOM_MESH,
                                              fileName=self.filename,
                                              meshScale=self.scale)
        body_id = p.createMultiBody(basePosition=self.pose['position'],
                                    baseOrientation=self.pose['orientation_quat'],
                                    baseMass=self._default_mass,
                                    baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=-1)
        return body_id


class Pedestrian(object):
    def __init__(self, style='standing', pos=[0, 0, 0]):
        self.collision_filename = os.path.join(gibson2.assets_path, 'models', 'person_meshes',
                                               'person_{}'.format(style), 'meshes',
                                               'person_vhacd.obj')
        self.visual_filename = os.path.join(gibson2.assets_path, 'models', 'person_meshes',
                                            'person_{}'.format(style), 'meshes', 'person.obj')
        self.body_id = None
        self.cid = None

        self.pos = pos

    def load(self):
        collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=self.collision_filename)
        visual_id = p.createVisualShape(p.GEOM_MESH, fileName=self.visual_filename)
        body_id = p.createMultiBody(basePosition=[0, 0, 0],
                                    baseMass=60,
                                    baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=visual_id)
        self.body_id = body_id

        p.resetBasePositionAndOrientation(self.body_id, self.pos, [-0.5, -0.5, -0.5, 0.5])

        self.cid = p.createConstraint(self.body_id,
                                      -1,
                                      -1,
                                      -1,
                                      p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                      self.pos,
                                      parentFrameOrientation=[-0.5, -0.5, -0.5,
                                                              0.5])    # facing x axis
        return body_id

    def reset_position_orientation(self, pos, orn):
        p.changeConstraint(self.cid, pos, orn)


class VisualObject(object):
    def __init__(self, visual_shape=p.GEOM_SPHERE, rgba_color=[1, 0, 0, 0.5], radius=1.0, half_extents=[1, 1, 1]):
        self.visual_shape = visual_shape
        self.rgba_color = rgba_color
        self.radius = radius
        self.half_extents = half_extents

    def load(self):
        if self.visual_shape == p.GEOM_BOX:
            shape = p.createVisualShape(self.visual_shape, rgbaColor=self.rgba_color, halfExtents=self.half_extents)
        else:
            shape = p.createVisualShape(self.visual_shape, rgbaColor=self.rgba_color, radius=self.radius)
        self.body_id = p.createMultiBody(baseVisualShapeIndex=shape, baseCollisionShapeIndex=-1)
        return self.body_id

    def set_position(self, pos, new_orn=None):
        if new_orn is None:
            _, new_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, new_orn)

    def set_color(self, color):
        p.changeVisualShape(self.body_id, -1, rgbaColor=color)


class BoxShape(object):
    def __init__(self, pos=[1, 2, 3], dim=[1, 2, 3]):
        self.basePos = pos
        self.dimension = dim

    def load(self):
        mass = 1000
        # basePosition = [1,2,2]
        baseOrientation = [0, 0, 0, 1]

        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.dimension)
        visualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=self.dimension)

        self.body_id = p.createMultiBody(baseMass=mass,
                                         baseCollisionShapeIndex=colBoxId,
                                         baseVisualShapeIndex=visualShapeId,
                                         basePosition=self.basePos,
                                         baseOrientation=baseOrientation)

        return self.body_id

    def set_position(self, pos):
        _, org_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, org_orn)


class InteractiveObj(object):
    def __init__(self, filename, scale=1):
        self.filename = filename
        self.scale = scale

    def load(self):
        self.body_id = p.loadURDF(self.filename, globalScaling=self.scale)
        return self.body_id

    def set_position(self, pos):
        org_pos, org_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, org_orn)

    def set_position_rotation(self, pos, orn):
        p.resetBasePositionAndOrientation(self.body_id, pos, orn)


class RBOObject(InteractiveObj):
    def __init__(self, name, scale=1):
        filename = os.path.join(gibson2.assets_path, 'models', 'rbo', name, 'configuration',
                                '{}.urdf'.format(name))
        super(RBOObject, self).__init__(filename, scale)
