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
        # config_file is 'str'   
        self.SR = 44100
        self.audio_system = None
        self.complete_audio_output = []
        super().__init__(config_file, scene_id, mode, automatic_reset=False)

            
    def load_observation_space(self):
        """
        Load observation space
        """
        self.output = self.config['output']
        self.image_width = self.config.get('image_width', 128)
        self.image_height = self.config.get('image_height', 128)
        observation_space = OrderedDict()
        sensors = OrderedDict()
        vision_modalities = []
        scan_modalities = []

        if 'task_obs' in self.output:
            observation_space['task_obs'] = self.build_obs_space(
                shape=(self.task.task_obs_dim,), low=-np.inf, high=np.inf)
        if 'rgb' in self.output:
            observation_space['rgb'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=0.0, high=1.0)
            vision_modalities.append('rgb')
        if 'depth' in self.output:
            observation_space['depth'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1),
                low=0.0, high=1.0)
            vision_modalities.append('depth')
        if 'scan' in self.output:
            self.n_horizontal_rays = self.config.get('n_horizontal_rays', 128)
            self.n_vertical_beams = self.config.get('n_vertical_beams', 1)
            assert self.n_vertical_beams == 1, 'scan can only handle one vertical beam for now'
            observation_space['scan'] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1),
                low=0.0, high=1.0)
            scan_modalities.append('scan')
        if 'occupancy_grid' in self.output:
            self.grid_resolution = self.config.get('grid_resolution', 128)
            self.occupancy_grid_space = gym.spaces.Box(low=0.0,
                                                       high=1.0,
                                                       shape=(self.grid_resolution,
                                                              self.grid_resolution, 1))
            observation_space['occupancy_grid'] = self.occupancy_grid_space
            scan_modalities.append('occupancy_grid')
        if 'bump' in self.output:
            observation_space['bump'] = gym.spaces.Box(low=0.0,
                                                       high=1.0,
                                                       shape=(1,))
            sensors['bump'] = BumpSensor(self)

        if len(vision_modalities) > 0:
            sensors['vision'] = VisionSensor(self, vision_modalities)

        if len(scan_modalities) > 0:
            sensors['scan_occ'] = ScanSensor(self, scan_modalities)

        self.observation_space = gym.spaces.Dict(observation_space)
        self.sensors = sensors

        
    def load(self):
        """
        Load scene, robot, and environment
        """
        if self.config['scene'] == 'gibson' or self.config['scene'] == 'mp3d':
            scene = StaticIndoorScene(
                self.config['scene_id'],
                waypoint_resolution=self.config.get(
                    'waypoint_resolution', 0.2),
                num_waypoints=self.config.get('num_waypoints', 10),
                build_graph=self.config.get('build_graph', False),
                trav_map_resolution=self.config.get(
                    'trav_map_resolution', 0.1),
                trav_map_erosion=self.config.get('trav_map_erosion', 2),
                pybullet_load_texture=self.config.get(
                    'pybullet_load_texture', False),
            )
            self.simulator.import_scene(
                scene, load_texture=self.config.get('load_texture', True))
        elif self.config['scene'] == 'igibson':
            scene = InteractiveIndoorScene(
                self.config['scene_id'],
                waypoint_resolution=self.config.get(
                    'waypoint_resolution', 0.2),
                num_waypoints=self.config.get('num_waypoints', 10),
                build_graph=self.config.get('build_graph', False),
                trav_map_resolution=self.config.get(
                    'trav_map_resolution', 0.1),
                trav_map_erosion=self.config.get('trav_map_erosion', 2),
                trav_map_type=self.config.get('trav_map_type', 'with_obj'),
                texture_randomization=self.texture_randomization_freq is not None,
                object_randomization=self.object_randomization_freq is not None,
                object_randomization_idx=self.object_randomization_idx,
                should_open_all_doors=self.config.get(
                    'should_open_all_doors', False),
                load_object_categories=self.config.get(
                    'load_object_categories', None),
                load_room_types=self.config.get('load_room_types', None),
                load_room_instances=self.config.get(
                    'load_room_instances', None),
            )
            
            first_n = self.config.get('_set_first_n_objects', -1)
            if first_n != -1:
                scene._set_first_n_objects(first_n)
            self.simulator.import_scene(scene)
            

        self.scene = scene
        robot_config = self.config["robot"]
        robot_name = robot_config.pop("name")
        robot = REGISTERED_ROBOTS[robot_name](**robot_config)
        self.robots = [robot]
        for robot in self.robots:
            self.simulator.import_robot(robot)
        
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()
        
        if self.config['scene'] == 'igibson':
            carpets = []
            if "carpet" in self.scene.objects_by_category.keys():
                carpets = self.scene.objects_by_category["carpet"]
            for carpet in carpets:
                for robot_link_id in range(p.getNumJoints(self.robots[0].get_body_ids()[0])):
                    for i in range(len(carpet.get_body_ids())):
                        p.setCollisionFilterPair(carpet.get_body_ids()[i], 
                                                 self.robots[0].get_body_ids()[0], -1, robot_link_id, 0)
    
    def get_state(self):
        """
        Get the current observation

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """
        state = super().get_state()
        current_output = self.audio_system.current_output.astype(np.float32, order='C') / 32768.0
        self.complete_audio_output.extend(current_output)
        return state
    
    
    def step(self, action):
        """
        Apply robot's action.
        Returns the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        
        if action is not None:
            self.robots[0].apply_action(action)
            
        collision_links = self.run_simulation()
        self.collision_links = collision_links
        self.collision_step += int(len(collision_links) > 0)

        state = self.get_state()
        info = {}
        reward, info = self.task.get_reward(
            self, collision_links, action, info)
        done, info = self.task.get_termination(
            self, collision_links, action, info)
        self.task.step(self)
        self.populate_info(info)

        if done and self.automatic_reset:
            info['last_observation'] = state                
            state = self.reset()
        return state, reward, done, info
    
    
    def reset(self, split, sound_file, train=True):
        """
        Reset episode
        """
        # called at the beginning of the training or done for an episode
        self.randomize_domain()
        # move robot away from the scene
        self.robots[0].set_position([100.0, 100.0, 100.0])
        
        self.task.reset(self)
        
        if self.audio_system is not None:
            self.audio_system.disconnect()
            del self.audio_system
            self.audio_system = None
        self.complete_audio_output = []
        
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
        self.simulator.attachAudioSystem(self.audio_system)

        self.audio_system.step()
        self.simulator.sync(force_sync=True)
        state = self.get_state()
        self.reset_variables()

    
 