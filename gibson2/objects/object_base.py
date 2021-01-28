import pybullet as p
import os
import gibson2
from gibson2.object_properties.factory import get_object_property_class
from gibson2.object_states.factory import get_object_state_instance


class Object(object):
    """
    Base Object class
    """

    def __init__(self):
        self.body_id = None
        self.loaded = False

        self.prepare_object_properties()

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

    def prepare_object_properties(self):
        self.properties_name = ['onTop', 'inside',
                                'nextTo', 'under', 'touching']
        # TODO: append more properties name based on object taxonomy
        self.properties_name += []

        self.properties = {}
        for prop_name in self.properties_name:
            self.properties[prop_name] = get_object_property_class(prop_name)

        self.state_names = set()
        for prop_name in self.properties:
            self.state_names.update(
                self.properties[prop_name].get_relevant_states())

        self.states = {}
        for state_name in self.state_names:
            self.states[state_name] = get_object_state_instance(
                state_name, self)
