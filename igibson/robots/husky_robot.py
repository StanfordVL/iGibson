import gym
import numpy as np

from igibson.robots.robot_locomotor import LocomotorRobot


class Husky(LocomotorRobot):
    """
    Husky robot
    Reference: https://clearpathrobotics.com/, http://wiki.ros.org/Robots/Husky
    Uses joint torque control
    """

    def __init__(self, config):
        self.config = config
        self.torque = config.get("torque", 0.03)
        LocomotorRobot.__init__(self,
                                "husky/husky.urdf",
                                action_dim=4,
                                torque_coef=2.5,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="torque")

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
        self.action_list = [[self.torque, self.torque, self.torque, self.torque],
                            [-self.torque, -self.torque, -
                                self.torque, -self.torque],
                            [self.torque, -self.torque, self.torque, -self.torque],
                            [-self.torque, self.torque, -self.torque, self.torque], [0, 0, 0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def steering_cost(self, action):
        """
        Deprecated code for reward computation
        """
        if not self.is_discrete:
            return 0
        if action == 2 or action == 3:
            return -0.1
        else:
            return 0

    def alive_bonus(self, z, pitch):
        """
        Deprecated code for reward computation
        """
        top_xyz = self.parts["top_bumper_link"].get_position()
        bottom_xyz = self.parts["base_link"].get_position()
        alive = top_xyz[2] > bottom_xyz[2]
        # 0.25 is central sphere rad, die if it scrapes the ground
        return +1 if alive else -100

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  # forward
            (ord('s'),): 1,  # backward
            (ord('d'),): 2,  # turn right
            (ord('a'),): 3,  # turn left
            (): 4
        }
