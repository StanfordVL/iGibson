import gym
import numpy as np

from igibson.robots.locomotion_robot import LocomotionRobot


class Freight(LocomotionRobot):
    """
    Freight Robot
    Reference: https://fetchrobotics.com/robotics-platforms/freight-base/
    Uses joint velocity control
    """

    def __init__(self, config, **kwargs):
        self.config = config
        LocomotionRobot.__init__(
            self,
            "fetch/freight.urdf",
            action_dim=2,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control="velocity",
            **kwargs
        )

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        if not self.normalize_robot_action:
            raise ValueError("discrete action only works with normalized action space")
        self.action_list = [
            [1, 1],
            [-1, -1],
            [0.5, -0.5],
            [-0.5, 0.5],
            [0, 0],
        ]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord("w"),): 0,  # forward
            (ord("s"),): 1,  # backward
            (ord("d"),): 2,  # turn right
            (ord("a"),): 3,  # turn left
            (): 4,  # stay still
        }
