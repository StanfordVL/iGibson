from abc import abstractmethod

import gym
import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult, quat2mat

from igibson.controllers import LocomotionController
from igibson.robots.locomotion_robot import LocomotionRobot
from igibson.utils.constants import SemanticClass
from igibson.utils.python_utils import assert_valid_key


class TwoWheelRobot(LocomotionRobot):
    """
    Robot that is is equipped with locomotive (navigational) capabilities, as defined by two wheels that can be used
    for differential drive (e.g.: Turtlebot).
    Provides common interface for a wide variety of robots.

    NOTE: controller_config should, at the minimum, contain:
        base: controller specifications for the controller to control this robot's base (locomotion).
            Should include:

            - name: Controller to create
            - <other kwargs> relevant to the controller being created. Note that all values will have default
                values specified, but setting these individual kwargs will override them
    """

    def __init__(
        self,
        control_freq=10.0,
        action_config=None,
        controller_config=None,
        base_name=None,
        scale=1.0,
        self_collision=False,
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
            None defaults to the base link name used in @model_file
        :param scale: int, scaling factor for model (default is 1)
        :param self_collision: bool, whether to enable self collision
        :param class_id: SemanticClass, semantic class this robot belongs to. Default is SemanticClass.ROBOTS.
        :param rendering_params: None or Dict[str, Any], If not None, should be keyword-mapped rendering options to set.
            See DEFAULT_RENDERING_PARAMS for the values passed by default.
        """
        # Run super init
        super().__init__(
            control_freq=control_freq,
            action_config=action_config,
            controller_config=controller_config,
            base_name=base_name,
            scale=scale,
            self_collision=self_collision,
            class_id=class_id,
            rendering_params=rendering_params,
        )

    def _validate_configuration(self):
        # Make sure base only has two indices (i.e.: two wheels for differential drive)
        assert len(self.base_control_idx) == 2, "Differential drive can only be used with robot with two base joints!"

        # run super
        super()._validate_configuration()

    def _create_discrete_action_space(self):
        # Set action list based on controller (joint or DD) used

        # We set straight velocity to be 50% of max velocity for the wheels
        max_wheel_joint_vels = self.control_limits["velocity"][1][self.base_control_idx]
        assert len(max_wheel_joint_vels) == 2, "TwoWheelRobot must only have two base (wheel) joints!"
        assert max_wheel_joint_vels[0] == max_wheel_joint_vels[1], "Both wheels must have the same max speed!"
        wheel_straight_vel = 0.5 * max_wheel_joint_vels[0]
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

        # Return this action space
        return gym.spaces.Box(len(self.action_list))

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # Use DifferentialDrive as default
        controllers["base"] = "DifferentialDriveController"

        return controllers

    @property
    def _default_base_differential_drive_controller_config(self):
        """
        :return: Dict[str, Any] Default differential drive controller config to
            control this robot's base.
        """
        # Calculate max linear, angular velocities -- make sure each wheel has same max values
        max_vels = [list(self._joints.values())[i].max_velocity for i in self.base_control_idx]
        assert max_vels[0] == max_vels[1], "Differential drive requires both wheel joints to have same max velocities!"
        max_lin_vel = max_vels[0] * self.wheel_radius
        max_ang_vel = max_lin_vel * 2.0 / self.wheel_axle_length

        return {
            "name": "DifferentialDriveController",
            "control_freq": self.control_freq,
            "wheel_radius": self.wheel_radius,
            "wheel_axle_length": self.wheel_axle_length,
            "control_limits": self.control_limits,
            "joint_idx": self.base_control_idx,
            "command_output_limits": ([-max_lin_vel, -max_ang_vel], [max_lin_vel, max_ang_vel]),  # (lin_vel, ang_vel)
        }

    @property
    def _default_controller_config(self):
        # Always run super method first
        cfg = super()._default_controller_config

        # Add differential drive option to base
        cfg["base"][
            self._default_base_differential_drive_controller_config["name"]
        ] = self._default_base_differential_drive_controller_config

        return cfg

    @property
    @abstractmethod
    def wheel_radius(self):
        """
        :return: float, radius of each wheel at the base, in metric units
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def wheel_axle_length(self):
        """
        :return: float, perpendicular distance between the robot's two wheels, in metric units
        """
        raise NotImplementedError
