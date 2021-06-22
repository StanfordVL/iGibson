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


import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import random
import time
from torch.utils.tensorboard import SummaryWriter
import shutil
import h5py


class Model(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 arm_action_size):
        super(Model, self).__init__()
        assert num_layers > 0
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for i in range(1, num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.arm_action_head = nn.Linear(hidden_size, arm_action_size)

    def forward(self, x):
        x = self.layers(x)
        arm_action = self.arm_action_head(x)
        return arm_action


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
                    'urdf_file': '{}_neurips_task_{}_{}_0_fixed_furniture'.format(scene_id, task, task_id),
                    'load_object_categories': ["coffee_table", "cauldron"],
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

    model = Model(input_size=44,
                  hidden_size=64,
                  num_layers=5,
                  arm_action_size=28,
                  )
    model = torch.nn.DataParallel(model).cuda()

    resume = '/home/fei/Development/gibsonv2/gibson2/debug/bc_results/ckpt/model_best.pth.tar'

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        #if args.gpu is None:
        checkpoint = torch.load(resume)
        #else:
        #    # Map model to be loaded to specified single gpu.
        #    loc = 'cuda:{}'.format(args.gpu)
        #    checkpoint = torch.load(args.resume, map_location=loc)
        #args.start_epoch = checkpoint['epoch']
        # best_l1 = checkpoint['best_l1']
        # if args.gpu is not None:
        #    # best_acc1 may be from a checkpoint from a different GPU
        #    best_l1 = best_l1.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        input_mean = checkpoint['input_mean']
        input_std = checkpoint['input_std']

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    env = BehaviorEnv(config_file=args.config,
                     mode=args.mode,
                     action_timestep=1.0 / 30.0,
                     physics_timestep=1.0 / 300.0)

    env.simulator.viewer.px = -1.1
    env.simulator.viewer.py = 1.0
    env.simulator.viewer.pz = 5.4
    env.simulator.viewer.view_direction = np.array([0.2, -0.2, -0.2])
    step_time_list = []
    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        env.reset()
        #env.robots[0].set_position_orientation([ 0.32115,   -1.41382001,  1.31316097],
        #                                       [1.05644993e-02, - 6.99956664e-03,  9.99919350e-01,  8.30828918e-04]
        #                                       )

        for i in range(100):  # 10 seconds

            # construct input
            left_hand_local_pos = env.robots[0].parts['left_hand'].local_pos
            left_hand_local_orn = env.robots[0].parts['left_hand'].local_orn
            right_hand_local_pos = env.robots[0].parts['right_hand'].local_pos
            right_hand_local_orn = env.robots[0].parts['right_hand'].local_orn
            left_hand_trigger_fraction = [env.robots[0].parts['left_hand'].trigger_fraction]
            right_hand_trigger_fraction = [env.robots[0].parts['right_hand'].trigger_fraction]

            proprioception = np.concatenate((left_hand_local_pos, left_hand_local_orn,
                                             right_hand_local_pos, right_hand_local_orn,
                                             left_hand_trigger_fraction, right_hand_trigger_fraction),
                                            )
            keys = ['1', '62', '8', '98']
            tracked_objects = ['floor.n.01_1', 'caldron.n.01_1', 'table.n.02_1', 'agent.n.01_1']
            task_obs = []

            for obj in tracked_objects:
                pos, orn = env.task.object_scope[obj].get_position_orientation()
                task_obs.append(np.array(pos))
                task_obs.append(np.array(orn))

            task_obs = np.concatenate(task_obs)
            task_obs[21 + 2] += 0.6
            #from IPython import embed;
            #embed()

            agent_input = np.concatenate((proprioception, task_obs))
            agent_input = (agent_input - input_mean) / (input_std + 1e-10)
            agent_input = agent_input[None, :].astype(np.float32)

            with torch.no_grad():
                pred_action = model(torch.from_numpy(agent_input)).cpu().numpy()[0]

            print(pred_action)

            action = np.zeros((28,))#env.action_space.sample()
            #action[:6] = pred_action[:6]
            action[12:18] = pred_action[12:18]
            action[20:26] = pred_action[20:26]
            if i < 5:
                action[19] = 1
                action[27] = 1

            state, reward, done, _ = env.step(action)
            time.sleep(0.05)
            print('reward', reward)
            if done:
                break
        print('Episode finished after {} timesteps, took {} seconds.'.format(
            env.current_step, time.time() - start))
    env.close()
