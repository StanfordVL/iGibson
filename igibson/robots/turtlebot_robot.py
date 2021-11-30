import os
from collections import OrderedDict

import gym
import numpy as np

import igibson
from igibson.controllers import ControlType
from igibson.robots.locomotion_robot import LocomotionRobot
from igibson.utils.constants import SemanticClass
from igibson.utils.python_utils import assert_valid_key


class Turtlebot(LocomotionRobot):
    """
    Turtlebot robot
    Reference: http://wiki.ros.org/Robots/TurtleBot
    Uses joint velocity control
    """

    def __init__(
        self,
        control_freq=10.0,
        action_config=None,
        controller_config=None,
        base_name=None,
        scale=1.0,
        self_collision=True,
        class_id=SemanticClass.ROBOTS,
        rendering_params=None,
    ):
        """
        :param control_freq: float, control frequency (in Hz) at which to control the robot
        :param action_config: None or Dict[str, ...], potentially nested dictionary mapping action settings
            to action-related values. Should, at the minimum, contain:
                type: one of {discrete, continuous} - what type of action space to use
                normalize: either {True, False} - whether to normalize inputted actions
            This will override any default values specified by this class.
        :param controller_config: None or Dict[str, ...], nested dictionary mapping controller name(s) to specific controller
            configurations for this robot. This will override any default values specified by this class.
        :param base_name: None or str, robot link name that will represent the entire robot's frame of reference. If not None,
            this should correspond to one of the link names found in this robot's corresponding URDF / MJCF file.
            None defaults to the first link name used in @model_file
        :param scale: int, scaling factor for model (default is 1)
        :param self_collision: bool, whether to enable self collision
        :param class_id: SemanticClass, semantic class this robot belongs to. Default is SemanticClass.ROBOTS.
        :param rendering_params: None or Dict[str, Any], If not None, should be keyword-mapped rendering options to set.
            See DEFAULT_RENDERING_PARAMS for the values passed by default.
        """
        # Initialize action list variable if we're using discrete actions
        self.keys_to_action = None

        # Run super init
        super().__init__(
            model_file=os.path.join(igibson.assets_path, "models/turtlebot/turtlebot.urdf"),
            control_freq=control_freq,
            action_config=action_config,
            controller_config=controller_config,
            base_name=base_name,
            scale=scale,
            self_collision=self_collision,
            class_id=class_id,
            rendering_params=rendering_params,
        )

    def _create_discrete_action_space(self):
        # Set action list based on controller used
        wheel_straight_vel = 1.0
        wheel_rotate_vel = 0.5
        if self.controller_config["base"]["name"] == "JointController":
            action_list = [
                [wheel_straight_vel, wheel_straight_vel],
                [-wheel_straight_vel, -wheel_straight_vel],
                [wheel_rotate_vel, -wheel_rotate_vel],
                [-wheel_rotate_vel, wheel_rotate_vel],
                [0, 0],
            ]
        else:
            # DifferentialDriveController
            lin_vel = wheel_straight_vel * self.wheel_radius
            ang_vel = wheel_rotate_vel * self.wheel_radius * 2.0 / self.wheel_axle_length
            action_list = [
                [lin_vel, 0],
                [-lin_vel, 0],
                [0, ang_vel],
                [0, -ang_vel],
                [0, 0],
            ]

        self.action_list = action_list

        # Setup keybind mappings for teleoperation
        self.setup_keys_to_action()

        # Return this action space
        return gym.spaces.Box(len(self.action_list))

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord("w"),): 0,  # forward
            (ord("s"),): 1,  # backward
            (ord("d"),): 2,  # turn right
            (ord("a"),): 3,  # turn left
            (): 4,  # stay still
        }

    def get_proprioception(self):
        # We only get velocity info
        return np.concatenate([self.base_link.get_linear_velocity(), self.base_link.get_angular_velocity()])

    @property
    def wheel_radius(self):
        """
        :return: float, radius of each Turtlebot's wheel at the base, in metric units
        """
        return 0.038

    @property
    def wheel_axle_length(self):
        """
        :return: float, perpendicular distance between the two wheels of the Turtlebot, in metric units
        """
        return 0.23

    @property
    def base_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([0, 1])

    @property
    def default_joint_pos(self):
        return np.zeros(self.n_joints)
