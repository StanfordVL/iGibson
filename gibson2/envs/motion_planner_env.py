from gibson2.core.physics.interactive_objects import VisualMarker, InteractiveObj, BoxShape
import gibson2
from gibson2.utils.utils import parse_config, rotate_vector_3d, l2_distance, quatToXYZW
from gibson2.envs.base_env import BaseEnv
from transforms3d.euler import euler2quat
from collections import OrderedDict
import argparse
from gibson2.learn.completion import CompletionNet, identity_init, Perceptual
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from transforms3d.quaternions import quat2mat, qmult
import gym
import numpy as np
import os
import pybullet as p
from IPython import embed
import cv2
import time
import collections
from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv
from gibson2.external.pybullet_tools.utils import plan_base_motion, set_base_values

class MotionPlanningEnv(NavigateRandomEnv):
    def __init__(self,
                 config_file,
                 model_id=None,
                 collision_reward_weight=0.0,
                 mode='headless',
                 action_timestep=1 / 10.0,
                 physics_timestep=1 / 240.0,
                 device_idx=0,
                 automatic_reset=False,
                 ):
        super(MotionPlanningEnv, self).__init__(config_file,
                                                           model_id=model_id,
                                                           mode=mode,
                                                           action_timestep=action_timestep,
                                                           physics_timestep=physics_timestep,
                                                           automatic_reset=automatic_reset,
                                                           random_height=False,
                                                           device_idx=device_idx)

    def prepare_motion_planner(self):
        self.robot_id = self.robots[0].robot_ids[0]
        self.mesh_id = self.scene.mesh_body_id
        self.map_size = self.scene.trav_map_original_size * self.scene.trav_map_default_resolution

        print(self.robot_id, self.mesh_id, self.map_size)

    def plan_base_motion(self, x,y,theta):
        half_size = self.map_size / 2.0
        path = plan_base_motion(self.robot_id, [x,y,theta], ((-half_size, -half_size), (half_size, half_size)), obstacles=[self.mesh_id], restarts=10,
                                iterations=50, smooth=30)
        return path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    
    args = parser.parse_args()

    nav_env = MotionPlanningEnv(config_file=args.config,
                          mode=args.mode,
                          action_timestep=1.0 / 10.0,
                          physics_timestep=1.0 / 40.0)

    nav_env.prepare_motion_planner()

    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        nav_env.reset()
        path = nav_env.plan_base_motion(nav_env.target_pos[0], nav_env.target_pos[1], 0)
        if path is not None:
            for bq in path:
                set_base_values(nav_env.robot_id, [bq[0], bq[1], bq[2]])
                nav_env.step([0,0])
                #time.sleep(0.05)
            #nav_env.step()
        #for step in range(50):  # 500 steps, 50s world time
        #    action = nav_env.action_space.sample()
        #    state, reward, done, _ = nav_env.step(action)
        #    # print('reward', reward)
        #    if done:
        #        print('Episode finished after {} timesteps'.format(step + 1))
        #        break
        print(time.time() - start)
    nav_env.clean()
