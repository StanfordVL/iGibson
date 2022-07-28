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
from torch import device
from utils import dataset
# from utils.utils import quaternion_from_coeff, cartesian_to_polar
from utils.dataset import CATEGORIES, CATEGORY_MAP
from utils.logs import logger
from transforms3d.euler import euler2quat


from igibson.audio.ig_acoustic_mesh import getIgAcousticMesh
from igibson.audio.matterport_acoustic_mesh import getMatterportAcousticMesh


COUNT_CURR_EPISODE = 0


class AVNavRLEnv(iGibsonEnv):
    """
    Redefine the environment (robot, task, dataset)
    """
    def __init__(self, config_file, mode, scene_splits, device_idx):
        scene_id = np.random.choice(scene_splits)
        super().__init__(config_file, scene_id, mode, device_idx=device_idx)
        self.config["scene_splits"] = scene_splits
        
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

