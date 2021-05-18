from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.task.task_base import iGTNTask
from gibson2.scenes.empty_scene import EmptyScene

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
        seed = 0,
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
        self.action_space = gym.spaces.Box(shape=(28,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

    def load_task_setup(self):
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
                    'urdf_file': '{}_task_{}_{}_0_fixed_furniture'.format(scene_id, task, task_id),
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

    def load_empty_scene(self):
        scene = EmptyScene()
        scene.objects_by_id = {}
        self.simulator.import_scene(scene, render_floor_plane=True)
        agent = BehaviorRobot(self.simulator)
        self.simulator.import_behavior_robot(agent)
        self.simulator.register_main_vr_robot(agent)
        self.robots = [agent]
        self.agent = agent
        self.simulator.robots.append(agent)
        self.task = types.SimpleNamespace()
        self.task.initial_state = p.saveState()
        self.task.reset_scene = lambda snapshot_id: p.restoreState(snapshot_id)
        self.task.check_success = lambda: (False, [])

    def load(self):
        """
        Load environment
        """
        if not self.config.get('debug', False):
            self.load_task_setup()
        else:
            self.load_empty_scene()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

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
        self.current_step += 1
        self.robots[0].update(action)

        state = self.get_state()
        info = {}
        done, satisfied_predicates = self.task.check_success()
        reward, info = self.get_reward(satisfied_predicates)
        self.simulator.step(self)
        info = { "satisfied_predicates": satisfied_predicates }
        
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

        return state
    def reset(self, resample_objects=False):
        """
        Reset episode
        """
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
                        choices=['headless', 'gui', 'iggui'],
                        default='gui',
                        help='which mode for simulation (default: headless)')
    args = parser.parse_args()

    env = BehaviorEnv(config_file=args.config,
                     mode=args.mode,
                     action_timestep=1.0 / 10.0,
                     physics_timestep=1.0 / 40.0)

    env.simulator.viewer.px = -1.1
    env.simulator.viewer.py = 1.0
    env.simulator.viewer.pz = 5.4
    env.simulator.viewer.view_direction = np.array([0.2, -0.2, -0.2])
    step_time_list = []
    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        env.reset()
        for i in range(1000):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            print('reward', reward)
            if done:
                break
        print('Episode finished after {} timesteps, took {} seconds.'.format(
            env.current_step, time.time() - start))
    env.close()
