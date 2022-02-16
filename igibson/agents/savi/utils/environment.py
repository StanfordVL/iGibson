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

        
class TimeReward(BaseRewardFunction):
    """
    Time reward
    A negative reward per time step
    """

    def __init__(self, config):
        super().__init__(config)
        self.time_reward_weight = self.config.get(
            'time_reward_weight', -0.01)

    def get_reward(self, task, env):
        """
        Reward is proportional to the number of steps
        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        return self.time_reward_weight
        
        
class SAViTask(PointNavRandomTask):
    # reward function
    def __init__(self, env):
        super().__init__(env)
        self.reward_funcions = [
            PotentialReward(self.config), # geodesic distance, potential_reward_weight
            PointGoalReward(self.config), # success_reward
            CollisionReward(self.config),
            TimeReward(self.config), # time_reward_weight
        ]
        
    def sample_initial_pose_and_target_pos(self, env):
        """
        Sample robot initial pose and target position

        :param env: environment instance
        :return: initial pose and target position
        """
        _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
        max_trials = 100
        dist = 0.0
        for _ in range(max_trials):
            cat = random.choice(CATEGORIES)
            scene_files = glob.glob(os.path.join(igibson.ig_dataset_path, "scenes", env.scene_id, f"urdf/{env.scene_id}.urdf"), 
                                   recursive=True)
            sf = scene_files[0]
            tree = ET.parse(sf)
            links = tree.findall(".//link[@category='%s']" % cat)
            if len(links) == 0:
                continue
            link = random.choice(links)
            joint_name = "j_"+link.attrib["name"]
            joint = tree.findall(".//joint[@name='%s']" % joint_name)
            target_pos = [float(i) for i in joint[0].find('origin').attrib["xyz"].split()]
            target_pos = np.array(target_pos)
            
            if env.scene.build_graph:
                _, dist = env.scene.get_shortest_path(
                    self.floor_num,
                    initial_pos[:2],
                    target_pos[:2], entire_path=False)
            else:
                dist = l2_distance(initial_pos, target_pos)
            if self.target_dist_min < dist < self.target_dist_max:
                env.cat = cat
                break
        if not (self.target_dist_min < dist < self.target_dist_max):
            print("WARNING: Failed to sample initial and target positions")
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        env.initial_pos = initial_pos
        env.initial_orn = initial_orn
        return initial_pos, initial_orn, target_pos
    
    

class AVNavRLEnv(iGibsonEnv):
    """
    Redefine the environment (robot, task, dataset)
    """
    def __init__(self, config_file, mode, scene_id='mJXqzFtmKg4'):
        
#         self.audio_obj = None
        self.SR = 44100
        self.audio_system = None
        self.audio_len = 4410
        self.time_len = 10
        self.audio_channel1 = np.zeros(self.audio_len*self.time_len)
        self.audio_channel2 = np.zeros(self.audio_len*self.time_len)
        
        self.cat = None # audio category
        self.initial_pos = None
        self.initial_orn = None
        self._episode_time = 0.0
        self.scene_id = scene_id
        super().__init__(config_file, scene_id, mode, automatic_reset=True)
        

    def load_task_setup(self):
        """
        Load task setup
        """
        super().load_task_setup()
#         if self.config['task'] == 'point_nav_AVNav':
#             self.task = PointNavAVNav(self)
#         elif self.config['task'] == 'SAVi':
#             self.task = SAViTask(self)
        self.task = SAViTask(self)  
            
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
        if 'category_belief' in self.output:
            observation_space['category_belief'] = self.build_obs_space(
                shape=(len(CATEGORIES),), low=0.0, high=1.0)
        if 'location_belief' in self.output:
            observation_space['location_belief'] = self.build_obs_space(
                shape=(2,), low=0.0, high=1.0)
        if 'pose_sensor' in self.output:
            observation_space['pose_sensor'] = self.build_obs_space(
                shape=(7,), low=-np.inf, high=np.inf)
        if 'category' in self.output:
            observation_space['category'] = self.build_obs_space(
                shape=(len(CATEGORIES),), low=0.0, high=1.0)
        if 'audio' in self.output:
            spectrogram, spectrogram_cat = \
                self.compute_spectrogram(np.ones(int(self.SR / (1 / self.simulator.render_timestep)) * 2))
            observation_space['audio'] = self.build_obs_space(
                shape=spectrogram.shape, low=-np.inf, high=np.inf)
            observation_space['audio_concat'] = self.build_obs_space(
                shape=spectrogram_cat.shape, low=-np.inf, high=np.inf)
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
        if 'pc' in self.output:
            observation_space['pc'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=-np.inf, high=np.inf)
            vision_modalities.append('pc')
        if 'optical_flow' in self.output:
            observation_space['optical_flow'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 2),
                low=-np.inf, high=np.inf)
            vision_modalities.append('optical_flow')
        if 'scene_flow' in self.output:
            observation_space['scene_flow'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=-np.inf, high=np.inf)
            vision_modalities.append('scene_flow')
        if 'normal' in self.output:
            observation_space['normal'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=-np.inf, high=np.inf)
            vision_modalities.append('normal')
        if 'seg' in self.output:
            observation_space['seg'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1),
                low=0.0, high=1.0)
            vision_modalities.append('seg')
        if 'rgb_filled' in self.output:  # use filler
            observation_space['rgb_filled'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=0.0, high=1.0)
            vision_modalities.append('rgb_filled')
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
        if self.config['scene'] == 'empty':
            scene = EmptyScene()
            self.simulator.import_scene(
                scene, load_texture=self.config.get('load_texture', True))
        elif self.config['scene'] == 'stadium':
            scene = StadiumScene()
            self.simulator.import_scene(
                scene, load_texture=self.config.get('load_texture', True))
        elif self.config['scene'] == 'gibson' or self.config['scene'] == 'mp3d':
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
#                 pybullet_load_texture=self.config.get(
#                     'pybullet_load_texture', False),
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
            # TODO: Unify the function import_scene and take out of the if-else clauses
            first_n = self.config.get('_set_first_n_objects', -1)
            if first_n != -1:
                scene._set_first_n_objects(first_n)
            self.simulator.import_scene(scene)
            

        robot_config = self.config["robot"]
        robot_name = robot_config.pop("name")
        robot = REGISTERED_ROBOTS[robot_name](**robot_config)
        
        self.scene = scene
        self.robots = [robot]
        for robot in self.robots:
            self.simulator.import_robot(robot)
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()
        
        
    def compute_spectrogram(self, audio_data):
        def compute_stft(signal):
            n_fft = 512
            hop_length = 160
            win_length = 400
            stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
#             stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
            return stft
        
        def compute_stft_cat(signal):
            n_fft = 512
            hop_length = 160
            win_length = 400
            stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
            stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
            return stft
        
        self.audio_channel1 = np.append(self.audio_channel1[self.audio_len:], audio_data[::2])
        self.audio_channel2 = np.append(self.audio_channel2[self.audio_len:], audio_data[1::2])
        channel1_magnitude_cat = np.log1p(compute_stft_cat(self.audio_channel1))
        channel2_magnitude_cat = np.log1p(compute_stft_cat(self.audio_channel2))
        
        channel1_magnitude = np.log1p(compute_stft(audio_data[::2]))
        channel2_magnitude = np.log1p(compute_stft(audio_data[1::2]))
        
        spectrogram = np.stack([channel1_magnitude, channel2_magnitude], axis=-1)
        spectrogram_cat = np.stack([channel1_magnitude_cat, channel2_magnitude_cat], axis=-1)

        return spectrogram, spectrogram_cat
    

    def get_state(self):
        """
        Get the current observation

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """
        state = super().get_state()
        if 'audio' in self.output:
            current_output = self.audio_system.current_output.astype(np.float32, order='C') / 32768.0
            state['audio'], state['audio_concat'] = self.compute_spectrogram(current_output)

        
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
    
    
    def step(self, action, i_env, train):
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
    
    
    def reset(self, train=True):
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
        
        if 'audio' in self.output:
            write_to_file = self.config.get('audio_write', "")
            
            if self.config['scene'] == 'gibson' or self.config['scene'] == 'mp3d':
                acousticMesh = getMatterportAcousticMesh(self.simulator, 
                              "/cvgl/group/Gibson/matterport3d-downsized/v2/"+self.config['scene_id']+"/sem_map.png")
            elif self.config['scene'] == 'igibson':
                acousticMesh = getIgAcousticMesh(self.simulator)

            occl_multiplier = self.config.get('occl_multiplier', default_audio_config.OCCLUSION_MULTIPLIER)
            self.audio_system = AudioSystem(self.simulator, self.robots[0], acousticMesh, 
                                          is_Viewer=False, writeToFile=write_to_file, SR = self.SR,
                                          occl_multiplier=occl_multiplier)

            source_location = self.task.target_pos
            self.audio_obj = cube.Cube(pos=source_location, dim=[0.05, 0.05, 0.05], 
                                       visual_only=False, 
                                       mass=0.5, color=[255, 0, 0, 1]) # pos initialized with default
            self.simulator.import_object(self.audio_obj)
            self.audio_obj_id = self.audio_obj.get_body_id()[0]
            # for savi
            if train:
                self.audio_system.registerSource(self.audio_obj_id, self.config['audio_dir'] \
                                             +"/train/"+self.cat+".wav", enabled=True)
            else:
                self.audio_system.registerSource(self.audio_obj_id, self.config['audio_dir'] \
                                             +"/val/"+self.cat+".wav", enabled=True)    
#             self.audio_system.setSourceRepeat(self.audio_obj_id)
            self.simulator.attachAudioSystem(self.audio_system)
            self.audio_channel1 = np.zeros(self.audio_len*self.time_len)
            self.audio_channel2 = np.zeros(self.audio_len*self.time_len)

            self.audio_system.step()
        self.simulator.sync(force_sync=True)
        state = self.get_state()
        self.reset_variables()

        return state
