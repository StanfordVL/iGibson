import pybullet as p
from gibson2 import assets
import os

class YCBObject:
    def __init__(self, name, scale=1):
        self.filename =  os.path.join(os.path.dirname(os.path.abspath(assets.__file__)), 'models', 'ycb', name, 'textured_simple.obj')
        self.scale = scale

    def load(self):
        collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=self.filename, meshScale=self.scale)
        body_id = p.createMultiBody(basePosition=[0, 0, 0], baseMass=0.1, baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=-1)
        return body_id


class RBOObject:
    def __init__(self, name, scale=1):
        self.filename = os.path.join(os.path.dirname(os.path.abspath(assets.__file__)), 'models', 'rbo', name, 'configuration', '{}.urdf'.format(name))
        self.scale = scale

    def load(self):
        body_id = p.loadURDF(self.filename, globalScaling=self.scale)
        return body_id
