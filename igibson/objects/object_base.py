import pybullet as p


class Object(object):
    """
    Base Object class
    """

    def __init__(self):
        self.body_id = None
        self.loaded = False

        # initialize with empty states
        self.states = dict()
        # handle to instances in the renderer
        self.renderer_instances = []

    def highlight(self):
        for instance in self.renderer_instances:
            instance.set_highlight(True)

    def unhighlight(self):
        for instance in self.renderer_instances:
            instance.set_highlight(False)

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

    def get_position_orientation(self):
        """
        Get object position and orientation

        :return: position in xyz
        :return: quaternion in xyzw
        """
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        return pos, orn

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

    def set_base_link_position_orientation(self, pos, orn):
        dynamics_info = p.getDynamicsInfo(self.body_id, -1)
        inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
        pos, orn = p.multiplyTransforms(pos, orn, inertial_pos, inertial_orn)
        self.set_position_orientation(pos, orn)

    def rotate_by(self, x=0, y=0, z=0):
        """
        Rotates an object by given euler angles
        """
        e_x, e_y, e_z = p.getEulerFromQuaternion(self.get_orientation())
        self.set_orientation(p.getQuaternionFromEuler([e_x + x, e_y + y, e_z + z]))

    def dump_state(self):
        """Dumps the state of the object other than what's not included in pybullet state."""
        return None

    def load_state(self, dump):
        """Loads the state of the object other than what's not included in pybullet state."""
        pass
