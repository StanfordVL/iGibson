import gym
import numpy as np

from igibson.robots.locomotion_robot import LocomotionRobot


class Ant(LocomotionRobot):
    """
    OpenAI Ant Robot
    Uses joint torque control
    """

    def __init__(self, config, **kwargs):
        self.config = config
        LocomotionRobot.__init__(
            self,
            "ant/ant.xml",
            action_dim=8,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control="torque",
            **kwargs
        )

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        assert False, "Ant does not support discrete actions"
