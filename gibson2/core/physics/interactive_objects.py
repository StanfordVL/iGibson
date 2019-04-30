import pybullet as p
from gibson2 import assets
import numpy as np
import os


class ShapeNetObject(object):
    def __init__(self, path, scale=1., position=[0, 0, 0], orientation=[0, 0, 0]):
        self.filename =  path
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
        collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=self.filename, meshScale=self.scale)
        body_id = p.createMultiBody(basePosition=self.pose['position'],
                                    baseOrientation=self.pose['orientation_quat'],
                                    baseMass=self._default_mass,
                                    baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=-1)
        return body_id

class YCBObject:
    def __init__(self, name, scale=1):
        self.filename =  os.path.join(os.path.dirname(os.path.abspath(assets.__file__)), 'models', 'ycb', name, 'textured_simple.obj')
        self.scale = scale

    def load(self):
        collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=self.filename, meshScale=self.scale)
        body_id = p.createMultiBody(basePosition=[0, 0, 0], baseMass=0.1, baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=-1)
        return body_id


class InteractiveObj:
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
        filename = os.path.join(os.path.dirname(os.path.abspath(assets.__file__)), 'models', 'rbo', name,
                                     'configuration', '{}.urdf'.format(name))
        super(RBOObject, self).__init__(filename, scale)

