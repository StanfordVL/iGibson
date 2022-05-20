import argparse
import logging
import os
import time
from collections import OrderedDict

import gym
# https://github.com/openai/gym/blob/master/gym/spaces/multi_discrete.py
from gym import spaces
import copy
import numpy as np
import pybullet as p
from transforms3d.euler import euler2quat
import random
import bddl
import yaml
import matplotlib.pyplot as plt

import bddl

import igibson
from igibson import object_states
from igibson.envs.env_base import BaseEnv
from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots.robot_base import BaseRobot
from igibson.sensors.bump_sensor import BumpSensor
from igibson.sensors.scan_sensor import ScanSensor
from igibson.sensors.vision_sensor import VisionSensor
from igibson.tasks.behavior_task import BehaviorTask
from igibson.tasks.dummy_task import DummyTask
from igibson.tasks.dynamic_nav_random_task import DynamicNavRandomTask
from igibson.tasks.interactive_nav_random_task import InteractiveNavRandomTask
from igibson.tasks.point_nav_fixed_task import PointNavFixedTask
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.tasks.reaching_random_task import ReachingRandomTask
from igibson.tasks.room_rearrangement_task import RoomRearrangementTask
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.constants import MAX_INSTANCE_COUNT, SemanticClass
from igibson.utils.utils import quatToXYZW
from igibson.utils.utils import parse_config
from igibson.utils.constants import ViewerMode
from igibson.object_states.pose import Pose
from igibson.utils.transform_utils import quat2mat, quat2axisangle, mat2euler
from igibson.utils.motion_planning_utils import MotionPlanner
from igibson.object_states.soaked import Soaked
from igibson.robots.manipulation_robot import IsGraspingState
from igibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, BaseActionPrimitiveSet

from igibson.action_primitives.b1k_discrete_action_primitives import B1KActionPrimitives
from igibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitive
from igibson.envs.action_primitive_env import ActionPrimitivesEnv

logger = logging.getLogger(__name__)

from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper

log = logging.getLogger(__name__)

index_action_mapping = {
    0: 'move',
    1: 'pick',
    2: 'place',
    3: 'toggle',
    4: 'pull',
    5: 'push',
    6: 'vis_pick',
    7: 'vis_place',
    8: 'vis_pull',
    9: 'vis_push',
    10: 'toggle',
}

class SkillEnv(gym.Env):
    """
    Skill RL Environment (OpenAI Gym interface).
    """

    def __init__(self, selection="user", headless=False, short_exec=False,
                 accum_reward_obs=False,
                 obj_joint_obs=False,
                 config_file="fetch_behavior_aps.yaml",
                 dense_reward=True,
                 ):
        config_filename = os.path.join(igibson.configs_path, config_file)
        self.config = parse_config(config_filename)
        # print(self.config['output'])
        self.env = ActionPrimitivesEnv(
            "B1KActionPrimitives",
            config_file=self.config,
            mode="headless" if headless else "gui_interactive",
            use_pb_gui=False,  # (not headless and platform.system() != "Darwin"),
            num_attempts=1,
        )
        self.env.task.initial_state = self.env.task.save_scene(self.env)
        self.reset()

        # env.env.simulator.viewer.initial_pos = [1.5, -2.0, 2.3]
        # env.env.simulator.viewer.initial_view_direction = [-0.7, 0.0, -0.6]
        self.env.env.simulator.viewer.initial_pos = [1.0, -0.3, 1.9]
        self.env.env.simulator.viewer.initial_view_direction = [-0.1, -0.8, -0.5]
        self.env.env.simulator.viewer.reset_viewer()

        # Observation Space
        self.accum_reward_obs = accum_reward_obs
        self.obj_joint_obs = obj_joint_obs
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.state = None
        self.reward = 0.
        self.done = False
        self.info = {}
        self.dense_reward = dense_reward
        self.accum_reward = np.array([0.])
        self.step_index = 0
        # self.default_obs = OrderedDict()

    def reset(self):
        self.state = self.env.reset()
        self.accum_reward = np.array([0.])
        if self.env.env.config["task"] in ["putting_away_Halloween_decorations"]:
            self.env.env.scene.open_all_objs_by_category(category="bottom_cabinet", mode="value", value=0.2)
            print("bottom_cabinet opened!")
        self.state['accum_reward'] = self.accum_reward
        self.step_index = 0
        # self.default_obs = OrderedDict()
        # # print(self.state)
        # for key, value in self.state.items():
        #     self.default_obs[key] = np.ones_like(value)
        # print('self.default_obs: ', self.default_obs)
        return self.state

    def close(self):
        self.env.close()

    def step(self, action_idx):
        o, r, d, i = self.env.step(action_idx)
        self.accum_reward = self.accum_reward + r
        if self.dense_reward:
            if self.config['task'] == 'installing_a_printer':
                r = r - 0.1
            else:  # in ['putting_away_Halloween_decorations']:
                r = r - 0.01
        self.state = o
        self.reward = r
        self.done = d
        self.info = i
        if d:
            i["is_success"] = i["success"]
            # print('is_success: {}'.format(i['is_success']))

        if i["primitive_success"]:
            print("Primitive success!")
        else:
            print("Primitive {} failed. Ending".format(action_idx))
        self.state['accum_reward'] = self.accum_reward
        print('self.accum_reward: ', self.state['accum_reward'])
        self.step_index = self.step_index + 1
        # print('\n\n\n\nself.step_index: ', self.step_index)
        # print(self.state, self.reward)
        return self.state, self.reward, self.done, self.info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default=os.path.join(igibson.configs_path, "fetch_rl.yaml"),
                        help="which config file to use [default: use yaml files in examples/configs]")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "headless_tensor", "gui_interactive", "gui_non_interactive"],
        default='gui_interactive', # 'gui_interactive',  # "headless",
        help="which mode for simulation (default: headless)",
    )
    args = parser.parse_args()

    env = SkillEnv()

    action_list = [0, 1, 2, 3, 0, 4, 5, 6, 0, 4, 7, ]
    # action_list_2 = [0, 1, 2, 3, 4, 5, 6, 7]
    # action_list = [0, 1]

    for episode in range(1):
        print("Episode: {}".format(episode))
        # env.reset()
        start = time.time()
        for action_idx in action_list:
            state, reward, done, info = env.step(action_idx)
            print("{}, reward: {}, done: {}, is_success: {}".format(action_idx, reward, done, info['success']))
        print("Episode finished after {} timesteps, took {} seconds.".format(len(action_list), time.time() - start))
    env.close()
