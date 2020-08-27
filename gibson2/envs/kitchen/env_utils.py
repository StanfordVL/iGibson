from collections import OrderedDict
import numpy as np
import pybullet as p
from copy import deepcopy

import gibson2.external.pybullet_tools.utils as PBU
import gibson2.external.pybullet_tools.transformations as T
import gibson2.envs.kitchen.transform_utils as TU
from gibson2.core.physics.interactive_objects import Object


class ObjectBank(object):
    """A thin wrapper for managing objects"""
    def __init__(self, objects=None):
        self._objects = OrderedDict()
        if objects is not None:
            self._objects = deepcopy(objects)

    def add_object(self, name, o):
        assert isinstance(name, str)
        assert name not in self._objects
        assert isinstance(o, Object)
        self._objects[name] = o

    @property
    def object_list(self):
        return list(self._objects.values())

    @property
    def objects(self):
        return self._objects

    @property
    def body_ids(self):
        return tuple([o.body_id for o in list(self._objects.values())])

    def body_id_to_name(self, body_id):
        return self.names[self.body_ids.index(body_id)]

    def name_to_body_id(self, name):
        return self.body_ids[self.names.index(name)]

    def __getitem__(self, name):
        return self._objects[name]

    def __len__(self):
        return len(self._objects)

    @property
    def names(self):
        return [k for k in self.objects]

    @property
    def all_body_ids_and_links(self):
        body_ids = []
        links = []
        for bid in self.body_ids:
            for link in PBU.get_all_links(bid):
                body_ids.append(bid)
                links.append(link)
        return body_ids, links

    def serialize(self):
        body_ids = []
        links = []
        poses = []
        joint_positions = []
        for bid, link in zip(*self.all_body_ids_and_links):
            body_ids.append(bid)
            links.append(link)
            if link == -1:
                joint_positions.append(0.)
            else:
                joint_positions.append(PBU.get_joint_position(bid, link))
            poses.append(pose_to_array(PBU.get_link_pose(bid, link)))
        return {
            "body_ids": np.array(body_ids, dtype=np.int32),
            "links": np.array(links, dtype=np.int32),
            "link_poses": np.array(poses, dtype=np.float32),
            "joint_positions": np.array(joint_positions, dtype=np.float32)
        }

    def deserialize(self, states):
        assert "link_poses" in states
        assert "joint_positions" in states

        body_ids, links = self.all_body_ids_and_links
        assert states["link_poses"].shape == (len(body_ids), 7)
        assert states["joint_positions"].shape == (len(body_ids),)

        for i, (bid, link) in enumerate(zip(body_ids, links)):
            if link == -1:
                pose = states["link_poses"][i]
                p.resetBasePositionAndOrientation(bid, pose[:3], pose[3:])
            else:
                p.resetJointState(bid, link, targetValue=states["joint_positions"][i])

    def create_visual_copy(self, scale=1., rgba_alpha=None):
        new_bank = self.__class__()
        for name, obj in self._objects.items():
            obj_cpy = create_virtual_copy_of(obj, scale=scale, rgba_alpha=rgba_alpha)
            new_bank.add_object(name, obj_cpy)
        return new_bank


class PlannerObjectBank(ObjectBank):
    def __init__(self, objects):
        super(PlannerObjectBank, self).__init__(objects)
        self.reference = None

    def synchronize(self, other=None):
        if other is None:
            other = self.reference
        assert isinstance(other, ObjectBank)
        self.deserialize(other.serialize())

    @classmethod
    def create_from(cls, other, scale=1., rgba_alpha=None):
        assert isinstance(other, ObjectBank)
        new_bank = cls(other.create_visual_copy(scale=scale, rgba_alpha=rgba_alpha).objects)
        new_bank.reference = other
        return new_bank

    def get_visual_copy_of(self, ref_bid):
        """Get visual of the referenced object"""
        return self[self.reference.body_id_to_name(ref_bid)]


def set_articulated_object_dynamics(obj_id):
    assert isinstance(obj_id, int)
    for jointIndex in PBU.get_joints(obj_id):
        friction = 0
        p.setJointMotorControl2(obj_id, jointIndex, p.POSITION_CONTROL, force=friction)
    set_friction(obj_id, friction=10.)


def set_friction(obj_id, friction=10.):
    assert isinstance(obj_id, int)
    for l in PBU.get_all_links(obj_id):
        p.changeDynamics(obj_id, l, lateralFriction=friction)


def set_collision_between(obj1_id, obj2_id, collision):
    """
    Set collision behavior between two objects
    Args:
        obj1_id (int): body id
        obj2_id (int): body id
        collision (int): 0 for no collision, 1 for collision
    """
    for l1 in PBU.get_all_links(obj1_id):
        for l2 in PBU.get_all_links(obj2_id):
            p.setCollisionFilterPair(obj1_id, obj2_id, l1, l2, collision)


def change_object_rgba(obj_id, rgba, links=None):
    """
    Change rgba color of an object
    Args:
        obj_id (int): body id
        rgba (tuple): (r, g, b, a) each in range [0, 1]
        links (tuple, list): links for which color should be changed. Set to None to change for all links
    """
    assert len(rgba) == 4
    if links is None:
        links = PBU.get_links(obj_id)
    for l in links:
        p.changeVisualShape(obj_id, l, rgbaColor=rgba)


def change_object_alpha(obj_id, alpha, links=None):
    assert isinstance(alpha, float)
    if links is None:
        links = PBU.get_links(obj_id)

    all_visuals = p.getVisualShapeData(obj_id)
    for vis in all_visuals:
        link_index = vis[1]  # link index
        rgba = vis[7]  # current rgba
        if link_index in links:
            new_rgba = tuple(rgba[:3]) + (alpha,)
            p.changeVisualShape(obj_id, link_index, rgbaColor=new_rgba)


def create_virtual_copy_of(obj, scale=1., rgba_alpha=None):
    """
    Create a virtual (non-collision) copy of a target object
    Args:
        obj: obj to be copied
        scale: scale of the virtual object
        rgba_alpha: optionally change the alpha channel of the object color
    """
    assert isinstance(obj, Object)
    all_body_ids = [i for i in range(p.getNumBodies())]
    obj_cpy = deepcopy(obj)
    obj_cpy.loaded = False
    if hasattr(obj_cpy, "scale"):
        obj_cpy.scale = scale
    obj_cpy.load()
    assert obj_cpy.body_id != obj.body_id
    # disable collision with all objects
    for bid in all_body_ids:
        set_collision_between(obj_cpy.body_id, bid, collision=0)

    # change alpha channel if applies
    if rgba_alpha is not None:
        assert isinstance(rgba_alpha, float)
        change_object_alpha(obj.body_id, alpha=rgba_alpha)
    return obj_cpy


def pose_to_array(pose):
    assert len(pose) == 2
    assert len(pose[0]) == 3
    assert len(pose[1]) == 4
    return np.hstack((pose[0], pose[1]))


def pose_to_action_euler(start_pose, target_pose, max_dpos, max_drot=None):
    action = np.zeros(6)
    action[:3] = target_pose[:3] - start_pose[:3]
    action[:3] = np.clip(action[:3] / max_dpos, -1., 1.)

    action[3:6] = T.euler_from_quaternion(T.quaternion_multiply(target_pose[3:], T.quaternion_inverse(start_pose[3:])))
    if max_drot is not None:
        action[3:] = np.clip(action[3:] / max_drot, -1., 1.)
    return action


def action_to_delta_pose_euler(action, max_dpos, max_drot=None):
    assert len(action) == 6
    delta_pos = action[:3] * max_dpos

    delta_euler = action[3:]
    if max_drot is not None:
        delta_euler *= max_drot

    delta_rot = T.quaternion_from_euler(*delta_euler)
    return delta_pos, delta_rot


def pose_to_action_axis_vector(start_pose, target_pose, max_dpos, max_drot=None):
    action = np.zeros(6)
    action[:3] = target_pose[:3] - start_pose[:3]
    action[:3] = np.clip(action[:3] / max_dpos, -1., 1.)
    if not np.allclose(target_pose[:3] - start_pose[:3], action[:3] * max_dpos):
        print("clipped position")

    delta_quat = T.quaternion_multiply(target_pose[3:], T.quaternion_inverse(start_pose[3:]))
    delta_axis, delta_angle = TU.quat2axisangle(delta_quat)
    delta_rotation = -TU.axisangle2vec(delta_axis, delta_angle)
    delta_rotation_cpy = delta_rotation.copy()
    if max_drot is not None:
        delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)
    action[3:] = delta_rotation

    if not np.allclose(delta_rotation_cpy, action[3:] * max_drot):
        print("clipped rotation")

    return action


def action_to_delta_pose_axis_vector(action, max_dpos, max_drot=None):
    assert len(action) == 6
    delta_pos = action[:3] * max_dpos

    delta_axis_vector = action[3:]
    if max_drot is not None:
        delta_axis_vector *= max_drot

    delta_rot = TU.axisangle2quat(*TU.vec2axisangle(-delta_axis_vector))
    return delta_pos, delta_rot


def objects_center_in_container(candidates, container_id, container_link=-1):
    contained = []
    container_aabb = PBU.get_aabb(container_id, container_link)
    for bid in candidates:
        if PBU.aabb_contains_point(p.getBasePositionAndOrientation(bid)[0], container_aabb):
            contained.append(bid)
    return contained
