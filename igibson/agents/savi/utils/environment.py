import gym
import numpy as np
import logging
import pybullet as p
import librosa
from skimage.measure import block_reduce
import random
from collections import OrderedDict
import time
import xml.etree.ElementTree as ET
import glob
import os

import igibson
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
from igibson.reward_functions.collision_reward import CollisionReward
from igibson.objects import cube
from igibson.audio.audio_system import AudioSystem
import igibson.audio.default_config as default_audio_config
from utils.logs import logger
from utils import dataset
# from utils.utils import quaternion_from_coeff, cartesian_to_polar
from utils.dataset import CATEGORIES, CATEGORY_MAP

from transforms3d.euler import euler2quat


from igibson.audio.ig_acoustic_mesh import getIgAcousticMesh
from igibson.audio.matterport_acoustic_mesh import getMatterportAcousticMesh


COUNT_CURR_EPISODE = 0

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
        self._episode_time = 0.0
        self.scene_id = scene_id
        super().__init__(config_file, scene_id, mode, automatic_reset=True)

    def load_observation_space(self):
        super().load_observation_space()
        spaces = self.observation_space.spaces.copy()

        if 'pose_sensor' in self.output:
            spaces['pose_sensor'] = self.build_obs_space(
                shape=(7,), low=-np.inf, high=np.inf)

        self.observation_space = gym.spaces.Dict(spaces)

    def get_state(self):
        """
        Get the current observation

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """
        state = super().get_state()
        
        if 'pose_sensor' in self.output:
            # pose sensor in the episode frame
            pos = self.robots[0].get_position() #[x,y,z]
            rpy = self.robots[0].get_rpy() #(3,)
            
            state['pose_sensor'] = np.array(
                [*pos, *rpy, self._episode_time],
                dtype=np.float32)
            self._episode_time += 1.0
        
        if 'category' in self.output:
            index = CATEGORY_MAP[self.cat]
            onehot = np.zeros(len(CATEGORIES))
            onehot[index] = 1
            state['category'] = onehot
        
        # categoty_belief and location_belief are updated in _collect_rollout_step
        if "category_belief" in self.output:
            state["category_belief"] = np.zeros(len(CATEGORIES))
        if "location_belief" in self.output:
            state["location_belief"] = np.zeros(2)

        return state
    

