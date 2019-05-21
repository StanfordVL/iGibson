import pybullet as p
import os
import gibson2

class YCBObject:
    def __init__(self, name, scale=1):
        self.filename = os.path.join(gibson2.assets_path, 'models', 'ycb', name, 'textured_simple.obj')
        self.scale = scale

    def load(self):
        collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=self.filename, meshScale=self.scale)
        body_id = p.createMultiBody(basePosition=[0, 0, 0], baseMass=0.1, baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=-1)
        return body_id


class VisualObject(object):
    def __init__(self, rgba_color=[1, 0, 0, 0.5], radius=1.0):
        self.rgba_color = rgba_color
        self.radius = radius

    def load(self):
        shape = p.createVisualShape(p.GEOM_SPHERE, rgbaColor=self.rgba_color, radius=self.radius)
        self.body_id = p.createMultiBody(baseVisualShapeIndex=shape,
                                         baseCollisionShapeIndex=-1)
        return self.body_id

    def set_position(self, pos):
        _, org_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, org_orn)



class CollisionObject(object):
    def __init__(self, pos=[1,2,3], dim=[1, 2, 3]):
        self.basePos = pos
        self.dimension = dim

    def load(self):
        mass = 200
        # basePosition = [1,2,2]
        baseOrientation = [0, 0, 0, 1]

        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.dimension)
        visualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=self.dimension)

        self.body_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=colBoxId, baseVisualShapeIndex=visualShapeId,
                                         basePosition=self.basePos, baseOrientation=baseOrientation)

        return self.body_id

    def set_position(self, pos):
        _, org_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, org_orn)


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
        filename = os.path.join(gibson2.assets_path, 'models', 'rbo', name,
                                     'configuration', '{}.urdf'.format(name))
        super(RBOObject, self).__init__(filename, scale)

