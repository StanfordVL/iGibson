import numpy as np
import pybullet as p

import igibson.utils.transform_utils as T
from igibson.utils.filters import MovingAverageFilter

# Different modes
JS_MODES = {
    "joint_space_position",
    "joint_space_velocity",
}


class JointSpaceController:
    """
    Simple arm_controller class to control towards a desired joint state
    We always resolve in joint velocities --> The output of the arm_controller is a velocity for each joint to reach
    the desired command

    Args:
        robot (BaseRobot): Robot to control
        config (dict): Config associated with this iG setup
    """

    def __init__(self, robot, config):
        # Store internal variables
        self.robot = robot
        self.config = config
        self.input_max = np.array(config["controller"]["input_max"])
        self.input_min = np.array(config["controller"]["input_min"])
        self.output_max = np.array(config["controller"]["output_max"])
        self.output_min = np.array(config["controller"]["output_min"])
        self.action_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
        self.action_output_transform = (self.output_max + self.output_min) / 2.0
        self.action_input_transform = (self.input_max + self.input_min) / 2.0
        self.lpf = MovingAverageFilter(obs_dim=len(self.robot.upper_joint_limits), filter_width=2)

    def reset(self):
        """
        Reset this arm_controller
        """
        self.lpf = MovingAverageFilter(obs_dim=len(self.robot.upper_joint_limits), filter_width=2)

    def scale_command(self, command):
        """
        Scales the inputted action based on internal limits

        Args:
            command (6-array): Inputted raw command

        Returns:
            6-array: Scaled command
        """
        # Clip command
        command = np.clip(command, self.input_min, self.input_max)
        return (command - self.action_input_transform) * self.action_scale + self.action_output_transform

    def get_current_error(self, current, set_point):
        """
        Returns an array of differences between the desired joint positions and current joint positions.
        Useful for PID control.

        :param current: the current joint positions
        :param set_point: the joint positions that are desired as a numpy array
        :return: the current error in the joint positions
        """
        error = current - set_point
        return error

    def control(self, command):
        """
        Execute IK control, given @command.

        Args:
            command (number of DoF-array): it can be desired joint positions, or velocities

        Returns:
            number of DoF-array: commanded joint velocities to achieve the inputted @command.
        """
        # First, scale command
        scaled_command = self.scale_command(command)

        if self.mode == "joint_positions":
            # Grab the resulting error and scale it by the velocity gain
            cmd_joint_vel = self.config["controller"]["kv_vel"] * self.get_current_error(
                current=self.robot.joint_position, set_point=scaled_command
            )
        elif self.mode == "joint_velocities":
            cmd_joint_vel = scaled_command

        # Return these commanded velocities
        return cmd_joint_vel
