import gym
import numpy as np

from igibson.robots.robot_locomotor import LocomotorRobot


class Ant(LocomotorRobot):
    """
    OpenAI Ant Robot
    Uses joint torque control
    """

    def __init__(self, config):
        self.config = config
        self.torque = config.get("torque", 1.0)
        LocomotorRobot.__init__(
            self,
            "ant/ant.xml",
            action_dim=8,
            torque_coef=2.5,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control="torque",
        )

    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.torque * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        assert False, "Ant does not support discrete actions"
