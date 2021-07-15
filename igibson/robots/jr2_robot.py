import gym
import numpy as np

from igibson.robots.robot_locomotor import LocomotorRobot


class JR2(LocomotorRobot):
    """
    JR2 robot (no arm)
    Reference: https://cvgl.stanford.edu/projects/jackrabbot/
    Uses joint velocity control
    """

    def __init__(self, config):
        self.config = config
        self.velocity = config.get('velocity', 1.0)
        LocomotorRobot.__init__(self,
                                "jr2_urdf/jr2.urdf",
                                action_dim=4,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", True),
                                control='velocity')

    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.velocity * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        self.action_list = [[self.velocity, self.velocity, 0, self.velocity],
                            [-self.velocity, -self.velocity, 0, -self.velocity],
                            [self.velocity, -self.velocity, -self.velocity, 0],
                            [-self.velocity, self.velocity, self.velocity, 0], [0, 0, 0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  # forward
            (ord('s'),): 1,  # backward
            (ord('d'),): 2,  # turn right
            (ord('a'),): 3,  # turn left
            (): 4
        }
