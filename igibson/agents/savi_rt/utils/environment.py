import gym
import numpy as np
import logging
import pybullet as p
import librosa
from skimage.measure import block_reduce
from collections import OrderedDict
import os
import random
import time
import glob
import xml.etree.ElementTree as ET

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots.turtlebot import Turtlebot
from igibson.robots.robot_base import BaseRobot
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
from utils.logs import logger
from utils import dataset
from utils.dataset import CATEGORIES, CATEGORY_MAP
from igibson.audio.ig_acoustic_mesh import getIgAcousticMesh
from igibson.audio.matterport_acoustic_mesh import getMatterportAcousticMesh
from igibson.utils.utils import rotate_vector_3d
from igibson.tasks.point_nav_savi_task import SAViTask
        

class AVNavRLEnv(iGibsonEnv):
    """
    Redefine the environment (robot, task, dataset)
    """
    def __init__(self, config_file, mode, scene_id='mJXqzFtmKg4'):
        super().__init__(config_file, scene_id, mode)
        
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