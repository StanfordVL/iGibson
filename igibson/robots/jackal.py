import os

import numpy as np

import igibson
from igibson.robots.locomotion_robot import LocomotionRobot


class Jackal(LocomotionRobot):
    """
    Jackal robot
    Reference: https://clearpathrobotics.com/, http://wiki.ros.org/Robots/Husky
    """

    @property
    def model_name(self):
        """
        :return str: robot model name
        """
        return "Jackal"

    def _create_discrete_action_space(self):
        raise ValueError("Jackal does not support discrete actions!")

    @property
    def base_control_idx(self):
        return np.array([0, 1, 2, 3])

    @property
    def default_joint_pos(self):
        return np.zeros(self.n_joints)

    @property
    def model_file(self):
        return os.path.join(igibson.assets_path, "models/jackal/jackal.urdf")
