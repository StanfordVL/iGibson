import os

import gym
import numpy as np

import igibson
from igibson.robots.two_wheel_robot import TwoWheelRobot
from igibson.utils.constants import SemanticClass


class Freight(TwoWheelRobot):
    """
    Freight Robot
    Reference: https://fetchrobotics.com/robotics-platforms/freight-base/
    Uses joint velocity control
    """

    def get_proprioception(self):
        # We only get velocity info
        return np.concatenate([self.base_link.get_linear_velocity(), self.base_link.get_angular_velocity()])

    @property
    def wheel_radius(self):
        return 0.0613

    @property
    def wheel_axle_length(self):
        return 0.372

    @property
    def base_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([0, 1])

    @property
    def default_joint_pos(self):
        return np.zeros(self.n_joints)

    @property
    def model_file(self):
        return os.path.join(igibson.assets_path, "models/fetch/freight.urdf")
