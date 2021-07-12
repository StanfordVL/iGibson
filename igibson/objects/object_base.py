import pybullet as p
import os
import igibson


class Object(object):
    """
    Base Object class
    """

    def __init__(self):
        self.body_id = None
        self.loaded = False

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
