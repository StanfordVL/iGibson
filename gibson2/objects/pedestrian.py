import os
import gibson2
from gibson2.objects.object_base import Object
import pybullet as p
import numpy as np


class Pedestrian(Object):
    """
    Pedestiran object
    """

    def __init__(self, style='standing', scale=1.0, visual_only=True):
        super(Pedestrian, self).__init__()
        self.collision_filename = os.path.join(
            gibson2.assets_path, 'models', 'person_meshes',
            'person_{}'.format(style), 'meshes', 'person_vhacd.obj')
        self.visual_filename = os.path.join(
            gibson2.assets_path, 'models', 'person_meshes',
            'person_{}'.format(style), 'meshes', 'person.obj')
        self.visual_only = visual_only
        self.scale = scale
        self.default_orn_euler = np.array([np.pi / 2.0, 0.0, np.pi / 2.0])

    def _load(self):
        """
        Load the object into pybullet
        """
        collision_id = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=self.collision_filename,
            meshScale=[self.scale] * 3)
        visual_id = p.createVisualShape(
            p.GEOM_MESH,
            fileName=self.visual_filename,
            meshScale=[self.scale] * 3)
        if self.visual_only:
            body_id = p.createMultiBody(baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=visual_id)
        else:
            body_id = p.createMultiBody(baseMass=60,
                                        baseCollisionShapeIndex=collision_id,
                                        baseVisualShapeIndex=visual_id)
        p.resetBasePositionAndOrientation(
            body_id,
            [0.0, 0.0, 0.0],
            p.getQuaternionFromEuler(self.default_orn_euler)
        )
        return body_id

    def set_yaw(self, yaw):
        euler_angle = [self.default_orn_euler[0],
                       self.default_orn_euler[1],
                       self.default_orn_euler[2] + yaw]
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(
            self.body_id, pos, p.getQuaternionFromEuler(euler_angle)
        )

    def get_yaw(self):
        quat_orientation = super().get_orientation()

        # Euler angles in radians ( roll, pitch, yaw )
        euler_orientation = p.getEulerFromQuaternion(quat_orientation)

        yaw = euler_orientation[2] - self.default_orn_euler[2]
        return yaw
