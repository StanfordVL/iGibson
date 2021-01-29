import pybullet as p
import os
import gibson2
from gibson2.object_states.factory import get_object_state_instance, get_default_state_names, \
    get_state_names_for_ability


class Object(object):
    """
    Base Object class
    """

    def __init__(self):
        self.body_id = None
        self.loaded = False

        self.prepare_object_states()

    def load(self):
        """
        Load the object into pybullet.
        _load() will be implemented in the subclasses
        """
        if self.loaded:
            return self.body_id
        self.body_id = self._load()
        self.loaded = True
        return self.body_id

    def get_position(self):
        """
        Get object position

        :return: position in xyz
        """
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return pos

    def get_orientation(self):
        """
        Get object orientation

        :return: quaternion in xyzw
        """
        _, orn = p.getBasePositionAndOrientation(self.body_id)
        return orn

    def set_position(self, pos):
        """
        Set object position

        :param pos: position in xyz
        """
        _, old_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, old_orn)

    def set_orientation(self, orn):
        """
        Set object orientation

        :param orn: quaternion in xyzw
        """
        old_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, old_pos, orn)

    def set_position_orientation(self, pos, orn):
        """
        Set object position and orientation
        :param pos: position in xyz
        :param orn: quaternion in xyzw
        """
        p.resetBasePositionAndOrientation(self.body_id, pos, orn)

    def rotate_by(self, x=0, y=0, z=0):
        """
        Rotates an object by given euler angles
        """
        e_x, e_y, e_z = p.getEulerFromQuaternion(self.get_orientation())
        self.set_orientation(p.getQuaternionFromEuler(
            [e_x + x, e_y + y, e_z + z]))

    def prepare_object_states(self):
        state_names = list(get_default_state_names())

        abilities = []  # TODO: get the object's ability names from taxonomy
        for ability in abilities:
            state_names.extend(get_state_names_for_ability(ability))

        self.states = dict()
        for state_name in state_names:
            self.states[state_name] = get_object_state_instance(state_name, self)

            # Add each state's dependencies, too
            for dependency in self.states[state_name].get_dependencies():
                if dependency not in state_names:
                    state_names.append(dependency)
