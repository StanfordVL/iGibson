from abc import abstractmethod

import gym
import numpy as np

from igibson.robots.locomotion_robot import LocomotionRobot


class FourWheelRobot(LocomotionRobot):
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

    #print("[four_wheel_robot::FourWheelRobot] START")
    #print("[four_wheel_robot::FourWheelRobot] END")

    def _validate_configuration(self):

        #print("[four_wheel_robot::FourWheelRobot::_validate_configuration] START")

        # Make sure base only has four indices
        assert len(self.base_control_idx) == 4, "[four_wheel_robot::FourWheelRobot::] ERROR: Differential drive can only be used with robot with four base joints!"

        # run super
        super()._validate_configuration()

        #print("[four_wheel_robot::FourWheelRobot::_validate_configuration] END")
    
    def _create_discrete_action_space(self):
        #print("[four_wheel_robot::FourWheelRobot::_create_discrete_action_space] START")
        # Set action list based on controller (joint or DD) used

        # We set straight velocity to be 50% of max velocity for the wheels
        max_wheel_joint_vels = self.control_limits["velocity"][1][self.base_control_idx]
        
        assert len(max_wheel_joint_vels) == 2, "FourWheelRobot must only have two base (wheel) joints!"
        assert max_wheel_joint_vels[0] == max_wheel_joint_vels[1], "All wheels must have the same max speed!"
        
        wheel_straight_vel = 1.0 * max_wheel_joint_vels[0]
        wheel_rotate_vel = 1.0
        
        if self.controller_config["base"]["name"] == "JointController":
            action_list = [
                [wheel_straight_vel, wheel_straight_vel, wheel_straight_vel, wheel_straight_vel],
                [-wheel_straight_vel, -wheel_straight_vel, -wheel_straight_vel, -wheel_straight_vel],
                [wheel_rotate_vel, -wheel_rotate_vel, wheel_rotate_vel, -wheel_rotate_vel],
                [-wheel_rotate_vel, wheel_rotate_vel, -wheel_rotate_vel, wheel_rotate_vel],
                [0, 0, 0, 0],
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

        #print("[four_wheel_robot::FourWheelRobot::_create_discrete_action_space] END")

        # Return this action space
        return gym.spaces.Discrete(len(self.action_list))

    def _get_proprioception_dict(self):
        #print("[four_wheel_robot::FourWheelRobot::_get_proprioception_dict] START")

        dic = super()._get_proprioception_dict()

        # Grab wheel joint velocity info
        joints = list(self._joints.values())
        wheel_joints = [joints[idx] for idx in self.base_control_idx]
        l_vel, r_vel, _, _ = [jnt.get_state()[1] for jnt in wheel_joints]

        # Compute linear and angular velocities
        lin_vel = (l_vel + r_vel) / 2.0 * self.wheel_radius
        ang_vel = (r_vel - l_vel) / self.wheel_axle_length

        # Add info
        dic["dd_base_lin_vel"] = np.array([lin_vel])
        dic["dd_base_ang_vel"] = np.array([ang_vel])

        #print("[four_wheel_robot::FourWheelRobot::_get_proprioception_dict] END")

        return dic

    @property
    def default_proprio_obs(self):
        obs_keys = super().default_proprio_obs
        return obs_keys + ["dd_base_lin_vel", "dd_base_ang_vel"]

    @property
    def _default_controllers(self):
        #print("[four_wheel_robot::FourWheelRobot::_default_controllers] START")

        # Always call super first
        controllers = super()._default_controllers

        # Use DifferentialDrive as default
        controllers["base"] = "DifferentialDriveController"

        #print("[four_wheel_robot::FourWheelRobot::_default_controllers] END")

        return controllers

    @property
    def _default_base_differential_drive_controller_config(self):
        """
        :return: Dict[str, Any] Default differential drive controller config to
            control this robot's base.
        """
        #print("[four_wheel_robot::FourWheelRobot::_default_base_differential_drive_controller_config] START")
        #print("[four_wheel_robot::FourWheelRobot::_default_base_differential_drive_controller_config] END")

        return {
            "name": "DifferentialDriveController",
            "control_freq": self.control_freq,
            "wheel_radius": self.wheel_radius,
            "wheel_axle_length": self.wheel_axle_length,
            "control_limits": self.control_limits,
            "joint_idx": self.base_control_idx,
        }

    @property
    def _default_controller_config(self):

        #print("[four_wheel_robot::FourWheelRobot::_default_controller_config] START")

        # Always run super method first
        cfg = super()._default_controller_config

        # Add differential drive option to base
        cfg["base"][
            self._default_base_differential_drive_controller_config["name"]
        ] = self._default_base_differential_drive_controller_config

        #print("[four_wheel_robot::FourWheelRobot::_default_controller_config] END")

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
