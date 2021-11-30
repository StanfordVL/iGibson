import gym
import numpy as np

from igibson.robots.locomotion_robot import LocomotionRobot


class Locobot(LocomotionRobot):
    """
    Locobot robot
    Reference: https://www.trossenrobotics.com/locobot-pyrobot-ros-rover.aspx
    Uses differentiable_drive / twist command control
    """

    def __init__(self, config, **kwargs):
        self.config = config
        # https://www.trossenrobotics.com/locobot-pyrobot-ros-rover.aspx
        # Maximum translational velocity: 70 cm/s
        # Maximum rotational velocity: 180 deg/s (>110 deg/s gyro performance will degrade)
        self.wheel_dim = 2
        self.wheel_axle_half = 0.115  # half of the distance between the wheels
        self.wheel_radius = 0.038  # radius of the wheels
        LocomotionRobot.__init__(
            self,
            "locobot/locobot.urdf",
            base_name="base_link",
            action_dim=self.wheel_dim,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control="differential_drive",
            **kwargs
        )

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        assert False, "Locobot does not support discrete actions"

    def get_end_effector_position(self):
        """
        Get end-effector position
        """
        return self.links["gripper_link"].get_position()
