import os

import numpy as np

import igibson
from igibson.robots.two_wheel_robot import TwoWheelRobot
from igibson.utils.constants import SemanticClass


class Locobot(TwoWheelRobot):
    """
    Locobot robot
    Reference: https://www.trossenrobotics.com/locobot-pyrobot-ros-rover.aspx
    """

    def get_proprioception(self):
        # We only get velocity info
        return np.concatenate([self.base_link.get_linear_velocity(), self.base_link.get_angular_velocity()])

    @property
    def wheel_radius(self):
        return 0.038

    @property
    def wheel_axle_length(self):
        return 0.230

    @property
    def base_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([1, 0])

    @property
    def default_joint_pos(self):
        return np.zeros(self.n_joints)

    @property
    def model_file(self):
        return os.path.join(igibson.assets_path, "models/locobot/locobot.urdf")
