from collections import OrderedDict
import numpy as np
import pybullet as p

import gibson2.external.pybullet_tools.utils as PBU
import gibson2.external.pybullet_tools.transformations as T
from gibson2.core.physics.interactive_objects import Object


def pose_to_array(pose):
    assert len(pose) == 2
    assert len(pose[0]) == 3
    assert len(pose[1]) == 4
    return np.hstack((pose[0], pose[1]))


def pose_to_action(start_pose, target_pose, max_dpos, max_drot=None):
    action = np.zeros(6)
    action[:3] = target_pose[:3] - start_pose[:3]
    action[3:6] = T.euler_from_quaternion(T.quaternion_multiply(target_pose[3:], T.quaternion_inverse(start_pose[3:])))
    action[:3] = np.clip(action[:3] / max_dpos, -1., 1.)
    if max_dpos is not None:
        action[3:] = np.clip(action[3:] / max_drot, -1., 1.)
    return action


class ObjectBank(object):
    def __init__(self):
        self._objects = OrderedDict()

    def add_object(self, name, o):
        assert isinstance(name, str)
        assert name not in self._objects
        assert isinstance(o, Object)
        self._objects[name] = o

    @property
    def objects(self):
        return list(self._objects.values())

    @property
    def body_ids(self):
        return tuple([o.body_id for o in list(self._objects.values())])

    def __getitem__(self, name):
        return self._objects[name]

    def __len__(self):
        return len(self._objects)


def set_articulated_object_dynamics(obj):
    for jointIndex in PBU.get_joints(obj.body_id):
        friction = 0
        p.setJointMotorControl2(obj.body_id, jointIndex, p.POSITION_CONTROL, force=friction)
    set_friction(obj, friction=10.)


def set_friction(obj, friction=10.):
    for l in PBU.get_all_links(obj.body_id):
        p.changeDynamics(obj.body_id, l, lateralFriction=friction)