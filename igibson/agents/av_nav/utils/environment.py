import gym
import numpy as np
import logging
import pybullet as p
import librosa
from skimage.measure import block_reduce

from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots import REGISTERED_ROBOTS
from igibson.robots.turtlebot import Turtlebot
from igibson.robots.robot_base import BaseRobot
from igibson.sensors.bump_sensor import BumpSensor
from igibson.sensors.scan_sensor import ScanSensor
from igibson.sensors.vision_sensor import VisionSensor
from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.reward_functions.potential_reward import PotentialReward
from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson.utils.utils import parse_config
from igibson.objects import cube
from igibson.audio.audio_system import AudioSystem
import igibson.audio.default_config as default_audio_config
from utils.logs import logger
from utils.dataset import dataset

from collections import OrderedDict
import time

from igibson.audio.ig_acoustic_mesh import getIgAcousticMesh
from igibson.audio.matterport_acoustic_mesh import getMatterportAcousticMesh


class AVNavTurtlebot(Turtlebot):
    """
    Redefine the robot
    """
    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        self.action_list = [[self.velocity, self.velocity], #[-self.velocity, -self.velocity],
                            [self.velocity * 0.5, -self.velocity * 0.5],
                            [-self.velocity * 0.5, self.velocity * 0.5], [0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()
        
    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  # forward
            (ord('d'),): 1,  # turn right
            (ord('a'),): 2,  # turn left
            (): 3  # stay still
        }

    

class AVNavRLEnv(iGibsonEnv):
    """
    Redefine the environment (robot, task, dataset)
    """
    def __init__(self, config_file, mode, scene_id='mJXqzFtmKg4'):
        super().__init__(config_file, scene_id, mode, automatic_reset=True)
        
    def load(self):
        """
        Load scene, robot, and environment
        """
        super().load()
        
        if self.config['scene'] == 'igibson':
            carpets = []
            if "carpet" in self.scene.objects_by_category.keys():
                carpets = self.scene.objects_by_category["carpet"]
            for carpet in carpets:
                for robot_link_id in range(p.getNumJoints(self.robots[0].get_body_ids()[0])):
                    for i in range(len(carpet.get_body_ids())):
                        p.setCollisionFilterPair(carpet.get_body_ids()[i], 
                                                 self.robots[0].get_body_ids()[0], -1, robot_link_id, 0)

    def step(self, action, i_env, train, scene_splits=None):
        return super().step(action)