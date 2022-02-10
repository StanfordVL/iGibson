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
    
    
class PointNavAVNav(PointNavRandomTask):
    """
    Redefine the task (reward functions)
    """
    def __init__(self, env):
        super().__init__(env)
        self.reward_functions = [
            PotentialReward(self.config), # geodesic distance, potential_reward_weight
            CollisionReward(self.config),
            PointGoalReward(self.config), # success_reward
            TimeReward(self.config), # time_reward_weight
        ]
    

class AVNavRLEnv(iGibsonEnv):
    """
    Redefine the environment (robot, task, dataset)
    """
    def __init__(self, config_file, mode, scene_id='mJXqzFtmKg4'):
        self.SR = 44100
        self.audio_system = None
        self.audio_len = 4410
        time_len = 2
        self.audio_channel1 = np.zeros(self.audio_len*time_len)
        self.audio_channel2 = np.zeros(self.audio_len*time_len)
        super().__init__(config_file, scene_id, mode, automatic_reset=True)
        

    def load_task_setup(self):
        """
        Load task setup
        """
        super().load_task_setup()
        self.task = PointNavAVNav(self)
        
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
        if 'audio' in self.output:
            spectrogram = self.compute_spectrogram(np.ones(int(self.SR / (1 / self.simulator.render_timestep)) * 2))
            observation_space['audio'] = self.build_obs_space(
                shape=spectrogram.shape, low=-np.inf, high=np.inf)
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
        
        if self.config['scene'] == 'igibson':
            carpets = []
            if "carpet" in self.scene.objects_by_category.keys():
                carpets = self.scene.objects_by_category["carpet"]
            for carpet in carpets:
                for robot_link_id in range(p.getNumJoints(self.robots[0].get_body_id())):
                    p.setCollisionFilterPair(carpet.get_body_id(), self.robots[0].get_body_id(), -1, robot_link_id, 0)
        
        
    def compute_spectrogram(self, audio_data):
        def compute_stft(signal):
            n_fft = 512
            hop_length = 160
            win_length = 400
            stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
#             stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
            return stft
        
        self.audio_channel1 = np.append(self.audio_channel1[self.audio_len:], audio_data[::2])
        self.audio_channel2 = np.append(self.audio_channel2[self.audio_len:], audio_data[1::2])
        channel1_magnitude = np.log1p(compute_stft(self.audio_channel1))
        channel2_magnitude = np.log1p(compute_stft(self.audio_channel2))
#         channel1_magnitude = np.log1p(compute_stft(audio_data[::2]))
#         channel2_magnitude = np.log1p(compute_stft(audio_data[1::2]))
        spectrogram = np.stack([channel1_magnitude, channel2_magnitude], axis=-1)

        return spectrogram

    
    def get_state(self):
        """
        Get the current observation

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """
        state = super().get_state()
        if 'audio' in self.output:
            current_output = self.audio_system.current_output.astype(np.float32, order='C') / 32768.0
            state['audio'] = self.compute_spectrogram(current_output)
        return state
    
    
    def step(self, action, i_env, train, scene_splits=None):
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
#         global COUNT_CURR_EPISODE
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
            
#             if COUNT_CURR_EPISODE == self.config['NUM_EPISODES_PER_SCENE']:
#                 if train:
#                     idx = np.random.randint(len(scene_splits[i_env]))
#                     next_scene_id = scene_splits[i_env][idx]
#                 else:
#                     next_scenes = dataset.SCENE_SPLITS['val']
#                     idx = np.random.randint(len(next_scenes))
#                     next_scene_id = next_scenes[idx]

#                 logger.info("reloading scene {} for env {}".format(next_scene_id, i_env))
#                 self.reload_model(next_scene_id) # disconnects the simulator
                
#                 COUNT_CURR_EPISODE = 0
                
            state = self.reset()
#             COUNT_CURR_EPISODE += 1

        return state, reward, done, info
    
    
    def reset(self):
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

            ## modified 1006 for mp3d audio
            if self.config['scene'] == 'gibson' or self.config['scene'] == 'mp3d':
                acousticMesh = getMatterportAcousticMesh(self.simulator, 
                              "/cvgl/group/Gibson/matterport3d-downsized/v2/"+self.config['scene_id']+"/sem_map.png")
            elif self.config['scene'] == 'igibson':
                acousticMesh = getIgAcousticMesh(self.simulator)
                
            occl_multiplier = self.config.get('occl_multiplier', default_audio_config.OCCLUSION_MULTIPLIER)
            self.audio_system = AudioSystem(self.simulator, self.robots[0], acousticMesh, 
                                          is_Viewer=False, writeToFile=write_to_file, SR = self.SR, occl_multiplier=occl_multiplier) 
    
            ## end modification   

            source_location = self.task.target_pos
            self.audio_obj = cube.Cube(pos=source_location, dim=[0.05, 0.05, 0.05], 
                                       visual_only=False, 
                                       mass=0.5, color=[255, 0, 0, 1]) # pos initialized with default
            self.simulator.import_object(self.audio_obj)
            self.audio_obj_id = self.audio_obj.get_body_ids()[0]
            self.audio_system.registerSource(self.audio_obj_id, self.config['audio_dir'], enabled=True)
            self.audio_system.setSourceRepeat(self.audio_obj_id)
            self.simulator.attachAudioSystem(self.audio_system)

            self.audio_system.step()
        self.simulator.sync(force_sync=True)
        state = self.get_state()
        self.reset_variables()

        return state    

    
    def test_valid_position(self, obj, pos, orn=None):
        """
        Test if the robot or the object can be placed with no collision

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :return: validity
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if is_robot else obj.body_id
        
        self.simulator.step(audio=False)
        collisions = list(p.getContactPoints(bodyA=body_id))

        if logging.root.level <= logging.DEBUG:  # Only going into this if it is for logging --> efficiency
            for item in collisions:
                logging.debug('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(
                    item[1], item[2], item[3], item[4]))

        has_collision = len(collisions) == 0

        return has_collision
    
    
    def land(self, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if is_robot else obj.body_id

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator.step(audio=False)
            if len(p.getContactPoints(bodyA=body_id)) > 0:
                land_success = True
                break

        if not land_success:
            print("WARNING: Failed to land")

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()