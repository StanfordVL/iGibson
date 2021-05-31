from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.task.task_base import iGTNTask
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
# for debugging
from gibson2.tasks.point_nav_random_task import PointNavRandomTask
from gibson2.tasks.reaching_random_task import ReachingRandomTask

import argparse
import numpy as np
import time
import tasknet
import types
import gym.spaces
import pybullet as p

from collections import OrderedDict
from gibson2.robots.behavior_robot import BehaviorRobot

class BehaviorEnv(iGibsonEnv):
    """
    iGibson Environment (OpenAI Gym interface)
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
        seed=0,
        action_filter='navigation'
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        self.action_filter = action_filter
        super(BehaviorEnv, self).__init__(config_file=config_file,
                                         scene_id=scene_id,
                                         mode=mode,
                                         action_timestep=action_timestep,
                                         physics_timestep=physics_timestep,
                                         device_idx=device_idx,
                                         render_to_tensor=render_to_tensor)
        self.rng = np.random.default_rng(seed=seed)
        self.automatic_reset = automatic_reset

    def load_action_space(self):
        """
        Load action space
        """
        if self.action_filter == 'navigation':
            self.action_space = gym.spaces.Box(shape=(3,),
                                               low=-1.0,
                                               high=1.0,
                                               dtype=np.float32)
        elif self.action_filter == 'mobile_manipulation':
            self.action_space = gym.spaces.Box(shape=(17,),
                                               low=-1.0,
                                               high=1.0,
                                               dtype=np.float32)
        elif self.action_filter == 'tabletop_manipulation':
            self.action_space = gym.spaces.Box(shape=(7,),
                                               low=-1.0,
                                               high=1.0,
                                               dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(shape=(28,),
                                               low=-1.0,
                                               high=1.0,
                                               dtype=np.float32)

    def load_behavior_task_setup(self):
        """
        Load task setup
        """
        self.initial_pos_z_offset = self.config.get(
            'initial_pos_z_offset', 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
        assert drop_distance < self.initial_pos_z_offset, \
            'initial_pos_z_offset is too small for collision checking'

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(
            self.config.get('collision_ignore_body_b_ids', []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(
            self.config.get('collision_ignore_link_a_ids', []))

        # discount factor
        self.discount_factor = self.config.get('discount_factor', 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get(
            'texture_randomization_freq', None)
        self.object_randomization_freq = self.config.get(
            'object_randomization_freq', None)

        # task
        task = self.config['task']
        task_id = self.config['task_id']
        scene_id = self.config['scene_id']
        clutter = self.config['clutter']
        online_sampling = self.config['online_sampling']
        if online_sampling:
            scene_kwargs = {}
        else:
            scene_kwargs = {
                    'urdf_file': '{}_neurips_task_{}_{}_0_fixed_furniture'.format(scene_id, task, task_id),
                    'load_object_categories': ["breakfast_table", "shelf", "swivel_chair", "notebook", "hardback"]
            }
        tasknet.set_backend("iGibson")
        self.task = iGTNTask(task, task_id)
        self.task.initialize_simulator(
                simulator=self.simulator, 
                scene_id=scene_id, 
                load_clutter=clutter, 
                scene_kwargs=scene_kwargs, 
                online_sampling=online_sampling
        )

        self.robots = [self.task.agent]

    def load_ig_task_setup(self):
        if self.config['scene'] == 'empty':
            scene = EmptyScene()
            scene.objects_by_id = {}
            self.simulator.import_scene(scene, render_floor_plane=True)
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
                pybullet_load_texture=self.config.get(
                    'pybullet_load_texture', False),
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
            self.simulator.import_ig_scene(scene)

        agent = BehaviorRobot(self.simulator, use_tracked_body_override=True, show_visual_head=True,
                              use_ghost_hands=False)
        self.simulator.import_behavior_robot(agent)
        self.simulator.register_main_vr_robot(agent)
        self.initial_pos_z_offset = 0.7

        self.robots = [agent]
        self.agent = agent
        self.simulator.robots.append(agent)
        self.scene = scene
        # task
        if self.config['task'] == 'point_nav_random':
            self.task = PointNavRandomTask(self)
        elif self.config['task'] == 'reaching_random':
            self.task = ReachingRandomTask(self)
        else:
            self.task = types.SimpleNamespace()
            self.task.initial_state = p.saveState()
            self.task.reset_scene = lambda snapshot_id: p.restoreState(snapshot_id)
            self.task.check_success = lambda: (False, [])

    def load_task_setup(self):
        pass

    def load(self):
        """
        Load environment
        """
        if not self.config.get('debug', False):
            self.load_behavior_task_setup()
        else:
            self.load_ig_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def load_observation_space(self):
        super(BehaviorEnv, self).load_observation_space()
        if 'proprioception' in self.output:
            proprioception_dim = self.robots[0].get_proprioception_dim()
            self.observation_space.spaces['proprioception'] = gym.spaces.Box(low=-100.0,
                                                       high=100.0,
                                                       shape=(proprioception_dim,))
            self.observation_space = gym.spaces.Dict(self.observation_space.spaces)

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
        if self.action_filter == 'navigation':
            action = action * 0.05
            new_action = np.zeros((28,))
            new_action[:2] = action[:2]
            new_action[5] = action[2]
        elif self.action_filter == 'mobile_manipulation':
            self.robots[0].hand_thresh = 0.8
            action = action * 0.05
            new_action = np.zeros((28,))
            # body x,y,yaw
            new_action[:2] = action[:2]
            new_action[5] = action[2]
            # left hand 7d
            new_action[12:19] = action[3:10]
            # right hand 7d
            new_action[20:27] = action[10:17]
        elif self.action_filter == 'tabletop_manipulation':
            # Note: only using right hand
            self.robots[0].hand_thresh = 0.8
            action = action * 0.05
            new_action = np.zeros((28,))
            new_action[20:27] = action[:7]
        else:
            new_action = action

        if self.current_step < 2:
            new_action[19] = 1
            new_action[27] = 1

        self.current_step += 1
        self.robots[0].update(new_action)

        state = self.get_state()
        info = {}
        if isinstance(self.task, PointNavRandomTask):
            reward, info = self.task.get_reward(self)
            done, done_info = self.task.get_termination(self)
            info.update(done_info)
            self.task.step(self)
        else:
            done, satisfied_predicates = self.task.check_success()
            reward, info = self.get_reward(satisfied_predicates)
            info = {"satisfied_predicates": satisfied_predicates}

        self.simulator.step(self)
        self.populate_info(info)

        if done and self.automatic_reset:
            info['last_observation'] = state
            state = self.reset()
        return state, reward, done, info

    @staticmethod
    def get_reward(satisfied_predicates):
        return satisfied_predicates, {}

    def get_state(self, collision_links=[]):
        """
        Get the current observation

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """
        state = OrderedDict()
        if 'task_obs' in self.output:
            state['task_obs'] = self.task.get_task_obs(self)
        if 'vision' in self.sensors:
            vision_obs = self.sensors['vision'].get_obs(self)
            for modality in vision_obs:
                state[modality] = vision_obs[modality]
        if 'scan_occ' in self.sensors:
            scan_obs = self.sensors['scan_occ'].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]
        if 'bump' in self.sensors:
            state['bump'] = self.sensors['bump'].get_obs(self)

        if 'proprioception' in self.output:
            state['proprioception'] = self.robots[0].get_proprioception()


        return state
    def reset(self, resample_objects=False):
        """
        Reset episode
        """
        self.robots[0].robot_specific_reset()

        if isinstance(self.task, PointNavRandomTask):
            self.task.reset_scene(self)
            self.task.reset_agent(self)
        else:
            self.task.reset_scene(snapshot_id=self.task.initial_state)
        self.simulator.sync()
        state = self.get_state()
        self.reset_variables()

        return state

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        default = 'gibson2/examples/configs/behavior.yaml',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui', 'pbgui'],
                        default='gui',
                        help='which mode for simulation (default: headless)')
    args = parser.parse_args()

    env = BehaviorEnv(config_file=args.config,
                      mode=args.mode,
                      action_timestep=1.0 / 10.0,
                      physics_timestep=1.0 / 40.0,
                      action_filter='mobile_manipulation')
    step_time_list = []
    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        env.reset()
        for i in range(1000):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            if done:
                break
        print('Episode finished after {} timesteps, took {} seconds.'.format(
            env.current_step, time.time() - start))
    env.close()
