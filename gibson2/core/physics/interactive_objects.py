import pybullet as p
import os
import gibson2
import numpy as np

class Object(object):
    def __init__(self):
        self.body_id = None
        self.loaded = False

    def load(self):
        return NotImplementedError()

    def get_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return pos

    def get_orientation(self):
        """Return object orientation
        :return: quaternion in xyzw
        """
        _, orn = p.getBasePositionAndOrientation(self.body_id)
        return orn

    def set_position(self, pos):
        _, old_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, old_orn)

    def set_orientation(self, orn):
        old_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, old_pos, orn)

    def set_position_orientation(self, pos, orn):
        p.resetBasePositionAndOrientation(self.body_id, pos, orn)


class YCBObject(Object):
    def __init__(self, name, scale=1):
        super(YCBObject, self).__init__()
        self.visual_filename = os.path.join(gibson2.assets_path, 'models', 'ycb', name,
                                     'textured_simple.obj')
        self.collision_filename = os.path.join(gibson2.assets_path, 'models', 'ycb', name,
                                     'textured_simple_vhacd.obj')
        self.scale = scale

    def load(self):
        if not self.loaded:
            collision_id = p.createCollisionShape(p.GEOM_MESH,
                                                  fileName=self.collision_filename,
                                                  meshScale=self.scale)
            visual_id = p.createVisualShape(p.GEOM_MESH,
                                                  fileName=self.visual_filename,
                                                  meshScale=self.scale)

            body_id = p.createMultiBody(baseCollisionShapeIndex=collision_id,
                                        baseVisualShapeIndex=visual_id,
                                        basePosition=[0.2, 0.2, 1.5],
                                        baseMass=0.1)

            self.body_id = body_id
            self.loaded = True
        return self.body_id


class ShapeNetObject(Object):
    def __init__(self, path, scale=1., position=[0, 0, 0], orientation=[0, 0, 0]):
        super(ShapeNetObject, self).__init__()
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
        if not self.loaded:
            collision_id = p.createCollisionShape(p.GEOM_MESH,
                                                  fileName=self.filename,
                                                  meshScale=self.scale)
            body_id = p.createMultiBody(basePosition=self.pose['position'],
                                        baseOrientation=self.pose['orientation_quat'],
                                        baseMass=self._default_mass,
                                        baseCollisionShapeIndex=collision_id,
                                        baseVisualShapeIndex=-1)
            self.body_id = body_id
            self.loaded = True

        return self.body_id


class Pedestrian(Object):
    def __init__(self, style='standing', pos=[0, 0, 0]):
        super(Pedestrian, self).__init__()
        self.collision_filename = os.path.join(gibson2.assets_path, 'models', 'person_meshes',
                                               'person_{}'.format(style), 'meshes',
                                               'person_vhacd.obj')
        self.visual_filename = os.path.join(gibson2.assets_path, 'models', 'person_meshes',
                                            'person_{}'.format(style), 'meshes', 'person.obj')
        self.cid = None
        self.pos = pos

    def load(self):
        if not self.loaded:
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
            self.loaded = True

        return self.body_id

    def reset_position_orientation(self, pos, orn):
        p.changeConstraint(self.cid, pos, orn)


class VisualMarker(Object):
    def __init__(self,
                 visual_shape=p.GEOM_SPHERE,
                 rgba_color=[1, 0, 0, 0.5],
                 radius=1.0,
                 half_extents=[1, 1, 1],
                 length=1,
                 initial_offset=[0, 0, 0]):
        """
        create a visual shape to show in pybullet and MeshRenderer

        :param visual_shape: pybullet.GEOM_BOX, pybullet.GEOM_CYLINDER, pybullet.GEOM_CAPSULE or pybullet.GEOM_SPHERE
        :param rgba_color: color
        :param radius: radius (for sphere)
        :param half_extents: parameters for pybullet.GEOM_BOX, pybullet.GEOM_CYLINDER or pybullet.GEOM_CAPSULE
        :param length: parameters for pybullet.GEOM_BOX, pybullet.GEOM_CYLINDER or pybullet.GEOM_CAPSULE
        :param initial_offset: visualFramePosition for the marker
        """
        super(VisualMarker, self).__init__()
        self.visual_shape = visual_shape
        self.rgba_color = rgba_color
        self.radius = radius
        self.half_extents = half_extents
        self.length = length
        self.initial_offset = initial_offset

    def load(self):
        if not self.loaded:
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
            self.loaded = True

        return self.body_id

    def set_position(self, pos, new_orn=None):
        if new_orn is None:
            _, new_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, new_orn)

    def set_color(self, color):
        p.changeVisualShape(self.body_id, -1, rgbaColor=color)


class BoxShape(Object):
    def __init__(self, pos=[1, 2, 3], dim=[1, 2, 3], visual_only=False, mass=1000, color=[1, 1, 1, 1]):
        super(BoxShape, self).__init__()
        self.basePos = pos
        self.dimension = dim
        self.visual_only = visual_only
        self.mass = mass
        self.color = color

    def load(self):
        if not self.loaded:
            baseOrientation = [0, 0, 0, 1]
            colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.dimension)
            visualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=self.dimension, rgbaColor=self.color)
            if self.visual_only:
                self.body_id = p.createMultiBody(baseCollisionShapeIndex=-1,
                                                 baseVisualShapeIndex=visualShapeId)
            else:
                self.body_id = p.createMultiBody(baseMass=self.mass,
                                                 baseCollisionShapeIndex=colBoxId,
                                                 baseVisualShapeIndex=visualShapeId)

            p.resetBasePositionAndOrientation(self.body_id, self.basePos, baseOrientation)
            self.loaded = True

        return self.body_id

    def set_position(self, pos):
        _, org_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, org_orn)


class InteractiveObj(Object):
    """
    Interactive Objects are represented as a urdf, but doesn't have motors
    """
    def __init__(self, filename, scale=1):
        super(InteractiveObj, self).__init__()
        self.filename = filename
        self.scale = scale

    def load(self):
        if not self.loaded:
            self.body_id = p.loadURDF(self.filename, globalScaling=self.scale, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
            self.mass = p.getDynamicsInfo(self.body_id, -1)[0]
            self.loaded = True

        return self.body_id

class SoftObject(Object):
    def __init__(self, filename, basePosition=[0,0,0], baseOrientation=[0,0,0,1], scale=-1, mass=-1, collisionMargin=-1, useMassSpring=0, useBendingSprings=0, useNeoHookean=0, springElasticStiffness=1, springDampingStiffness=0.1, springBendingStiffness=0.1, NeoHookeanMu=1, NeoHookeanLambda=1, NeoHookeanDamping=0.1, frictionCoeff=0, useFaceContact=0, useSelfCollision=0):
        super(SoftObject, self).__init__()
        self.filename = filename
        self.scale = scale
        self.basePosition = basePosition
        self.baseOrientation = baseOrientation
        self.mass = mass
        self.collisionMargin = collisionMargin
        self.useMassSpring = useMassSpring
        self.useBendingSprings = useBendingSprings
        self.useNeoHookean = useNeoHookean
        self.springElasticStiffness = springElasticStiffness
        self.springDampingStiffness = springDampingStiffness
        self.springBendingStiffness = springBendingStiffness
        self.NeoHookeanMu = NeoHookeanMu
        self.NeoHookeanLambda = NeoHookeanLambda
        self.NeoHookeanDamping = NeoHookeanDamping
        self.frictionCoeff = frictionCoeff
        self.useFaceContact = useFaceContact
        self.useSelfCollision = useSelfCollision

    def load(self):
        if not self.loaded:
            self.body_id = p.loadSoftBody(self.filename, scale = self.scale, basePosition = self.basePosition, baseOrientation = self.baseOrientation, mass=self.mass, collisionMargin=self.collisionMargin, useMassSpring=self.useMassSpring, useBendingSprings=self.useBendingSprings, useNeoHookean=self.useNeoHookean, springElasticStiffness=self.springElasticStiffness, springDampingStiffness=self.springDampingStiffness, springBendingStiffness=self.springBendingStiffness, NeoHookeanMu=self.NeoHookeanMu, NeoHookeanLambda=self.NeoHookeanLambda, NeoHookeanDamping=self.NeoHookeanDamping, frictionCoeff=self.frictionCoeff, useFaceContact=self.useFaceContact, useSelfCollision=self.useSelfCollision)

            # Set signed distance function voxel size (integrate to Simulator class?)
            p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.1)
            self.loaded = True

        return self.body_id

    def addAnchor(self, nodeIndex=-1, bodyUniqueId=-1, linkIndex=-1, bodyFramePosition=[0,0,0], physicsClientId=0):
        p.createSoftBodyAnchor(self.body_id, nodeIndex, bodyUniqueId, linkIndex, bodyFramePosition, physicsClientId)


class RBOObject(InteractiveObj):
    def __init__(self, name, scale=1):
        filename = os.path.join(gibson2.assets_path, 'models', 'rbo', name, 'configuration',
                                '{}.urdf'.format(name))
        super(RBOObject, self).__init__(filename, scale)
