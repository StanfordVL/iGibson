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
from igibson.sensors.scan_sensor import ScanSensor
from igibson.sensors.vision_sensor import VisionSensor
from igibson.sensors.bump_sensor import BumpSensor
from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.reward_functions.potential_reward import PotentialReward
from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson.reward_functions.collision_reward import CollisionReward

from igibson.utils.utils import parse_config
from igibson.objects import cube
from igibson.audio.audio_system import AudioSystem
from igibson.agents.savi.utils.logs import logger
from igibson.agents.savi.utils import dataset

from collections import OrderedDict
import time

from igibson.audio.ig_acoustic_mesh import getIgAcousticMesh
from igibson.audio.matterport_acoustic_mesh import getMatterportAcousticMesh



class AVNavRLEnv(iGibsonEnv):
    """
    Redefine the environment (robot, task, dataset)
    """
    def __init__(self, config_file, mode, scene_id='mJXqzFtmKg4'):
        super().__init__(config_file, scene_id, mode, automatic_reset=False)

        
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
    
    def reset(self, split, sound_file, train=True):
        """
        Reset episode
        """
        super().reset()

        if self.config['scene'] == 'gibson' or self.config['scene'] == 'mp3d':
            acousticMesh = getMatterportAcousticMesh(self.simulator, 
                          "/cvgl/group/Gibson/matterport3d-downsized/v2/"+self.config['scene_id']+"/sem_map.png")
        elif self.config['scene'] == 'igibson':
            acousticMesh = getIgAcousticMesh(self.simulator)

        self.audio_system = AudioSystem(self.simulator, self.robots[0], acousticMesh, 
                                      is_Viewer=False, writeToFile=self.config['audio_write']) 

        source_location = self.task.target_pos
        self.audio_obj = cube.Cube(pos=source_location, dim=[0.05, 0.05, 0.05], 
                                   visual_only=False, 
                                   mass=0.5, color=[255, 0, 0, 1]) # pos initialized with default
        self.simulator.import_object(self.audio_obj)
        self.audio_obj_id = self.audio_obj.get_body_ids()[0]
        self.audio_system.registerSource(self.audio_obj_id, 
                                         "/viscam/u/wangzz/avGibson/igibson/audio/semantic_splits/"
                                           +split+"/"+sound_file, 
                                         enabled=True)
#         self.audio_system.setSourceRepeat(self.audio_obj_id)

        self.audio_system.step()
        self.simulator.sync(force_sync=True)
        state = self.get_state()
        self.reset_variables()

    
 