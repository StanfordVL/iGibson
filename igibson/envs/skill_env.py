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

skill_object_offset_params = {
    0:  # skill id: move
        {
            'printer.n.03_1': [-0.7, 0, 0, 0],  # dx, dy, dz, target_yaw
            'table.n.02_1': [0, -0.6, 0, 0.5 * np.pi],
            # Pomaria_1_int, 2
            'hamburger.n.01_1': [0, -0.8, 0, 0.5 * np.pi],
            'hamburger.n.01_2': [0, -0.7, 0, 0.5 * np.pi],
            'hamburger.n.01_3': [0, -0.8, 0, 0.5 * np.pi],
            'ashcan.n.01_1': [0, 0.8, 0, -0.5 * np.pi],
            'countertop.n.01_1': [0.0, -0.8, 0, 0.5 * np.pi],  # [0.1, 0.5, 0.8 1.0]
            # 'countertop.n.01_1': [[0.0, -0.8, 0, 0.1 * np.pi], [0.0, -0.8, 0, 0.5 * np.pi], [0.0, -0.8, 0, 0.8 * np.pi],],  # [0.1, 0.5, 0.8 1.0]
            # # Ihlen_1_int, 0
            # 'hamburger.n.01_1': [0, 0.8, 0, -0.5 * np.pi],
            # 'hamburger.n.01_2': [0, 0.8, 0, -0.5 * np.pi],
            # 'hamburger.n.01_3': [-0.2, 0.7, 0, -0.6 * np.pi],
            # 'ashcan.n.01_1': [-0.2, -0.5, 0, 0.4 * np.pi],
            # 'countertop.n.01_1': [-0.5, -0.6, 0, 0.5 * np.pi],
            # putting_away_Halloween_decorations
            'pumpkin.n.02_1': [0.4, 0.0, 0.0, 1.0 * np.pi],
            'pumpkin.n.02_2': [0, -0.5, 0, 0.5 * np.pi],
            'cabinet.n.01_1': [0.4, -1.15, 0, 0.5 * np.pi],
            # cleaning_microwave_oven
            'sink.n.01_1-cleaning_microwave_oven': [0., -0.5, 0, 0.5 * np.pi],
            'microwave.n.02_1-cleaning_microwave_oven': [0., -1.0, 0, 0.5 * np.pi],
            'cabinet.n.01_1-cleaning_microwave_oven': [1.0, -0.8, 0, 0.5 * np.pi],
         },
    1: # pick
        {
            'printer.n.03_1': [-0.2, 0.0, 0.2],  # dx, dy, dz
            # Pomaria_1_int, 2
            'hamburger.n.01_1': [0.0, 0.0, 0.025],
            'hamburger.n.01_2': [0.0, 0.0, 0.025,],
            'hamburger.n.01_3': [0.0, 0.0, 0.025,],
            # putting_away_Halloween_decorations
            'pumpkin.n.02_1': [0.0, 0.0, 0.025,],
            'pumpkin.n.02_2': [0.0, 0.0, 0.025,],
            # cleaning_microwave_oven
            # 'rag.n.01_1': [0.0, 0.0, 0.025,],
            'towel': [0.0, 0.0, 0.0],
            # 'towel@cabinet.n.01_1-cleaning_microwave_oven': [1.0, -0.2, 0.5,],
            # # 'towel@sink.n.01_1-cleaning_microwave_oven': [-0.23, -0.1, 0.052],
            # 'towel@sink.n.01_1-cleaning_microwave_oven': [0.0, 0.0, 0.0],
        },
    2:  # place
        {
            'table.n.02_1': [0, 0, 0.5],  # dx, dy, dz
            # Pomaria_1_int, 2
            # 'ashcan.n.01_1': [0, 0, 0.5],
            # Ihlen_1_int, 0
            'ashcan.n.01_1': [0, 0, 0.5],
            # putting_away_Halloween_decorations
            # 'cabinet.n.01_1': [0.3, -0.55, 0.25],
            'cabinet.n.01_1': [0.3, -0.60, 0.25],
            # cleaning_microwave_oven
            'sink.n.01_1-cleaning_microwave_oven': [-0.2, -0.08, 0.2],
            'microwave.n.02_1-cleaning_microwave_oven': [-0.2, -0., -0.13],
        },
    3: # toggle
        {
            'printer.n.03_1': [-0.3, -0.25, 0.23],  # dx, dy, dz
            # cleaning_microwave_oven
            'sink.n.01_1-cleaning_microwave_oven': [-0.05, 0.18, 0.32],  #
            # 'sink.n.01_1-cleaning_microwave_oven': [-0.15, 0.2, 0.25],  #
        },
    4:  # pull
        {
            'cabinet.n.01_1': [0.3, -0.55, 0.35],  # dx, dy, dz
        },
    5:  # push
        {
            'cabinet.n.01_1': [0.3, -0.8, 0.35],  # dx, dy, dz
        },
    6:  # vis_pick
        {
            'hamburger.n.01_1': [0, -0.8, 0, 0.5 * np.pi, 0.0, 0.0, 0.025],
            'hamburger.n.01_2': [0, -0.7, 0, 0.5 * np.pi, 0.0, 0.0, 0.025,],
            'hamburger.n.01_3': [0, -0.8, 0, 0.5 * np.pi, 0.0, 0.0, 0.025,],
            # vis: putting_away_Halloween_decorations
            'pumpkin.n.02_1': [0.4, 0.0, 0.0, 1.0 * np.pi, 0.0, 0.0, 0.025,],
            'pumpkin.n.02_2': [0, -0.5, 0, 0.5 * np.pi, 0.0, 0.0, 0.025,],
        },
    7:  # vis_place
        {
            'ashcan.n.01_1': [0, 0.8, 0, -0.5 * np.pi, 0, 0, 0.5],
            # vis: putting_away_Halloween_decorations
            'cabinet.n.01_1': [0.4, -1.15, 0, 0.5 * np.pi, 0.3, -0.60, 0.25],
        },
    8:  # vis pull
        {
            'cabinet.n.01_1': [0.3, -0.55, 0.35],  # dx, dy, dz
        },
    9:  # vis push
        {
            'cabinet.n.01_1': [0.3, -0.8, 0.35],  # dx, dy, dz
        },
}

action_list_installing_a_printer = [
    [0, 'printer.n.03_1'],  # skill id, target_obj
    [1, 'printer.n.03_1'],
    [0, 'table.n.02_1'],
    [2, 'table.n.02_1'],
    [3, 'printer.n.03_1'],
]

# action_list_throwing_away_leftovers = [
#     [0, 'hamburger.n.01_1'],
#     [1, 'hamburger.n.01_1'],
#     [0, 'ashcan.n.01_1'],
#     [2, 'ashcan.n.01_1'],  # place
#     [0, 'hamburger.n.01_2'],
#     [1, 'hamburger.n.01_2'],
#     [0, 'ashcan.n.01_1'],
#     [2, 'ashcan.n.01_1'],  # place
#     [0, 'hamburger.n.01_3'],
#     [1, 'hamburger.n.01_3'],
#     [0, 'ashcan.n.01_1'],
#     [2, 'ashcan.n.01_1'],  # place
# ]*4

action_list_throwing_away_leftovers_v1 = [
    [0, 'hamburger.n.01_1'],
    [1, 'hamburger.n.01_1'],
    [0, 'ashcan.n.01_1'],
    [2, 'ashcan.n.01_1'],  # place
    [0, 'hamburger.n.01_2'],
    [1, 'hamburger.n.01_2'],
    [0, 'hamburger.n.01_3'],
    [1, 'hamburger.n.01_3'],
]

action_list_throwing_away_leftovers_discrete = [
    [0, 'countertop.n.01_1', 0],
    [0, 'countertop.n.01_1', 1],
    [0, 'countertop.n.01_1', 2],
    [6, 'hamburger.n.01_1'],
    [0, 'ashcan.n.01_1'],
    [7, 'ashcan.n.01_1'],  # place
    [6, 'hamburger.n.01_2'],
    [6, 'hamburger.n.01_3'],
]

action_list_throwing_away_leftovers = [
    [0, 'countertop.n.01_1', 0],
    [6, 'hamburger.n.01_1'],
    [0, 'ashcan.n.01_1'],
    [7, 'ashcan.n.01_1'],  # place
    [6, 'hamburger.n.01_2'],
    [6, 'hamburger.n.01_3'],
]

# action_list_throwing_away_leftovers = [
#     [0, 'countertop.n.01_1', 0],
#     [6, 'hamburger.n.01_2'],  # 1: 137, 2: 138, 3: 139, plate: 135, ashcan: 140
#     [0, 'ashcan.n.01_1'],
#     [7, 'ashcan.n.01_1'],  # place
#     [0, 'countertop.n.01_1', 1],
#     [6, 'hamburger.n.01_1'],
#     [0, 'ashcan.n.01_1'],
#     [7, 'ashcan.n.01_1'],  # place
#     [0, 'countertop.n.01_1', 2],
#     [6, 'hamburger.n.01_3'],
#     [0, 'ashcan.n.01_1'],
#     [7, 'ashcan.n.01_1'],  # place
# ]

# full set
# action_list_putting_away_Halloween_decorations = [
#     [0, 'cabinet.n.01_1'],  # move
#     [4, 'cabinet.n.01_1'],  # pull
#     [0, 'pumpkin.n.02_1'],  # move
#     [1, 'pumpkin.n.02_1'],  # pick
#     [2, 'cabinet.n.01_1'],  # place
#     [0, 'pumpkin.n.02_2'],  # move
#     [1, 'pumpkin.n.02_2'],  # pick
#     [5, 'cabinet.n.01_1'],  # push
# ]

# full sequence
action_list_putting_away_Halloween_decorations_v1 = [
    [0, 'cabinet.n.01_1'],  # move
    [4, 'cabinet.n.01_1'],  # pull
    [0, 'pumpkin.n.02_1'],  # move
    [1, 'pumpkin.n.02_1'],  # pick
    # [0, 'cabinet.n.01_1'],  # move
    [2, 'cabinet.n.01_1'],  # place
    [0, 'pumpkin.n.02_2'],  # move
    [1, 'pumpkin.n.02_2'],  # pick
    # [0, 'cabinet.n.01_1'],  # move
    # [2, 'cabinet.n.01_1'],  # place
    [5, 'cabinet.n.01_1'],  # push
]

# /home/robot/Desktop/behavior/iGibson-dev-jk/igibson/examples/robots/log_dir_his/20220510-001432_putting_away_Halloween_decorations_discrete_rgb_accumReward_m0.01
# wo vis operation
action_list_putting_away_Halloween_decorations_v2 = [
    [0, 'cabinet.n.01_1'],  # move
    [4, 'cabinet.n.01_1'],  # pull
    [0, 'pumpkin.n.02_1'],  # move
    [1, 'pumpkin.n.02_1'],  # pick
    [0, 'cabinet.n.01_1'],  # move, repeated
    [2, 'cabinet.n.01_1'],  # place
    [0, 'pumpkin.n.02_2'],  # move
    [1, 'pumpkin.n.02_2'],  # pick
    #
    # [0, 'cabinet.n.01_1'],  # move
    # [2, 'cabinet.n.01_1'],  # place
    #
    [5, 'cabinet.n.01_1'],  # push
] # * 4

# vis version: full sequence
action_list_putting_away_Halloween_decorations_v3 = [
    [0, 'cabinet.n.01_1'],  # move
    [4, 'cabinet.n.01_1'],  # vis pull
    [0, 'pumpkin.n.02_1'],  # move
    [6, 'pumpkin.n.02_1'],  # vis pick
    [0, 'cabinet.n.01_1'],  # move
    [7, 'cabinet.n.01_1'],  # vis place
    [0, 'pumpkin.n.02_2'],  # move
    [6, 'pumpkin.n.02_2'],  # vis pick
    [0, 'cabinet.n.01_1'],  # move
    [7, 'cabinet.n.01_1'],  # vis place
    [5, 'cabinet.n.01_1'],  # vis push
]
# vis version: full set
action_list_putting_away_Halloween_decorations = [
    [0, 'cabinet.n.01_1'],  # move
    [4, 'cabinet.n.01_1'],  # vis pull
    [0, 'pumpkin.n.02_1'],  # move
    [6, 'pumpkin.n.02_1'],  # vis pick
    # [0, 'cabinet.n.01_1'],  # move
    [7, 'cabinet.n.01_1'],  # vis place
    [0, 'pumpkin.n.02_2'],  # move
    [6, 'pumpkin.n.02_2'],  # vis pick
    # [0, 'cabinet.n.01_1'],  # move
    # [7, 'cabinet.n.01_1'],  # vis place
    [5, 'cabinet.n.01_1'],  # vis push
]

action_list_cleaning_microwave_oven = [
    [0, 'cabinet.n.01_1'],
    [1, 'towel'],  # pick
    [0, 'sink.n.01_1'],  # move
    # [2, 'sink.n.01_1'],  # place
    # [3, 'sink.n.01_1'],  # toggle
    # [1, 'towel'],  # pick
    # [0, 'microwave.n.02_1'],  # move
    # [2, 'microwave.n.02_1'],  # place
]

action_dict = {'installing_a_printer': action_list_installing_a_printer,
               'throwing_away_leftovers': action_list_throwing_away_leftovers,
               'cleaning_microwave_oven': action_list_cleaning_microwave_oven,
               'putting_away_Halloween_decorations': action_list_putting_away_Halloween_decorations,
               'throwing_away_leftovers_discrete': action_list_throwing_away_leftovers_discrete,}

class SkillEnv(gym.Env):
    """
    Skill RL Environment (OpenAI Gym interface).
    """

    def __init__(
            self,
            config_file,
            scene_id=None,
            mode="headless",
            action_timestep=1 / 60.0,
            physics_timestep=1 / 120.0,
            rendering_settings=None,
            vr_settings=None,
            device_idx=0,
            automatic_reset=False,
            use_pb_gui=False,
            print_log=True,
            dense_reward=True,
            action_space_type='discrete',
            multi_discrete_grid_size=10,
            accum_reward_obs=True,
            obj_joint_obs=False,
            rgb_obs=False,
            ins_seg_obs=True,
            is_success_count=True,
            obj_pose_check=False,
            visualize_arm_path=True,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, headless_tensor, gui_interactive, gui_non_interactive, vr
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param rendering_settings: rendering_settings to override the default one
        :param vr_settings: vr_settings to override the default one
        :param device_idx: which GPU to run the simulation and rendering on
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param use_pb_gui: concurrently display the interactive pybullet gui (for debugging)
        """
        self.print_log = print_log
        self.dense_reward = dense_reward
        self.accum_reward_obs = accum_reward_obs
        self.obj_joint_obs = obj_joint_obs
        self.is_success_count = is_success_count
        self.config = parse_config(config_file)
        self.action_space_type = action_space_type
        if self.config['task'] == 'throwing_away_leftovers' and self.action_space_type == 'discrete':
            self.action_list = action_dict['throwing_away_leftovers_discrete']
            skill_object_offset_params[0]['countertop.n.01_1'] = [[0.0, -0.8, 0, 0.1 * np.pi], [0.0, -0.8, 0, 0.5 * np.pi], [0.0, -0.8, 0, 0.8 * np.pi],]
        else:
            self.action_list = action_dict[self.config['task']]
        # print(self.config['task'] == 'throwing_away_leftovers', self.action_space_type == 'discrete', )
        # print(self.action_list)
        # print(skill_object_offset_params[0]['countertop.n.01_1'])
        self.num_discrete_action = len(self.action_list)
        self.automatic_reset = automatic_reset

        config_data = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
        config_data["hide_robot"] = False

        full_observability_2d_planning = True
        collision_with_pb_2d_planning = True
        self.step_counter = 0
        self.env = iGibsonEnv(
            config_file=config_data,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
        )
        # 'occupancy_grid' modality is required as input
        self.planner = MotionPlanner(
            self.env,
            optimize_iter=10,
            full_observability_2d_planning=full_observability_2d_planning,
            collision_with_pb_2d_planning=collision_with_pb_2d_planning,
            visualize_2d_planning=not (mode == 'headless') and False,
            visualize_2d_result=not (mode == 'headless') and False,
            fine_motion_plan=False,
            # print_log=print_log,
        )
        self.current_step = self.env.current_step

        self.env.task.initial_state = self.env.task.save_scene(self.env)

        if mode == 'gui_interactive':
            # Set viewer camera and directly set mode to planning mode
            self.env.simulator.viewer.initial_pos = [1.5, -1.0, 1.9]
            self.env.simulator.viewer.initial_view_direction = [-0.5, 0.0, -1.0]
            self.env.simulator.viewer.reset_viewer()
            self.env.simulator.viewer.mode = ViewerMode.PLANNING

        self.task_obj_list = self.env.task.object_scope
        print('self.task_obj_list: ', self.task_obj_list)
        self.observation_space = self.env.observation_space
        # print('self.observation_space: ', self.observation_space)
        if 'state_vec' in self.config["output"]:
            self.observation_space['state'] = spaces.Box(low=-1.0, high=2.0, shape=(1, 4), dtype=np.float32)
        if self.accum_reward_obs:
            self.observation_space['accum_reward'] = spaces.Box(low=-4.0, high=2.0, shape=(1, 1), dtype=np.float32)
        if self.obj_joint_obs:
            self.observation_space['obj_joint'] = spaces.Box(low=-4.0, high=2.0, shape=(1, 1), dtype=np.float32)
        if not rgb_obs:
            observation_space = type(self.observation_space)(
                [
                    (name, copy.deepcopy(space))
                    for name, space in self.observation_space.spaces.items()
                    if name not in ['rgb']
                ]
            )
            self.observation_space = observation_space
            # self.observation_space.pop('rgb', None)
        if not ins_seg_obs:
            observation_space = type(self.observation_space)(
                [
                    (name, copy.deepcopy(space))
                    for name, space in self.observation_space.spaces.items()
                    if name not in ['ins_seg']
                ]
            )
            self.observation_space = observation_space
            # self.observation_space.pop('ins_seg', None)
        self.sensors = self.env.sensors
        print('self.action_space_type: ', self.action_space_type)
        if self.action_space_type == 'multi_discrete':
            if self.config['task'] in ['throwing_away_leftovers']:
                multi_discrete_grid_size = 10
                self.action_space = spaces.MultiDiscrete([self.num_discrete_action] + [multi_discrete_grid_size, ] * 1)
            elif self.config['task'] in ['putting_away_Halloween_decorations']:
                multi_discrete_grid_size = 4
                self.action_space = spaces.MultiDiscrete([self.num_discrete_action] + [multi_discrete_grid_size, ] * 1)
        elif self.action_space_type == 'continuous':
            # self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -0.7 * np.pi]), high=np.array([1.0, 1.0, 1.0, 1.0, 0.7 * np.pi]), dtype=np.float32)
            self.action_space = spaces.Box(low=np.array([-1.0] * self.num_discrete_action + [-1.0, -1.0, -1.0, -1.0]),
                                           high=np.array([1.0] * self.num_discrete_action + [1.0, 1.0, 1.0, 1.0]), dtype=np.float32)

        else:  # 'discrete'
            self.action_space = spaces.Discrete(self.num_discrete_action)
        self.state = None
        self.accum_reward = np.array([0.])
        self.obj_joint = np.array([0.])
        self.is_success_list = []
        self.initial_pos_dict = {}
        self.multi_discrete_grid_size = multi_discrete_grid_size
        self.obj_pose_check = obj_pose_check
        self.soak_bonus = True
        self.visualize_arm_path_flag = visualize_arm_path

    def step(self, action_index):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: 0: move_to, 1: pick 2: place
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        self.step_counter += 1
        if self.action_space_type == 'multi_discrete':
            action = self.action_list[action_index[0]]
            params = skill_object_offset_params[action[0]][action[1]]
            if self.config['task'] in ['throwing_away_leftovers']:
                if action[0] in [0, ] and action[1] == 'countertop.n.01_1':
                    # print('params, new_params: ', params, new_params)
                    # [1, 10] * [1, pi]
                    params[3] = (action_index[1] + 1) * 0.1 * np.pi
                    print('action: {} {}, discrete action: {}, params: {}'.format(index_action_mapping[action[0]], action[1], action_index[1] + 1, params))
            elif self.config['task'] in ['putting_away_Halloween_decorations']:
                if action[0] in [0, ] and action[1] in ['pumpkin.n.02_1', 'pumpkin.n.02_2']:
                    # print('params, new_params: ', params, new_params)
                    # [1, 4] * [0.5*pi, 2 * pi]
                    # For debugging
                    if action_index[0] == 2:
                        action_index[1] = 1
                    elif action_index[0] == 6:
                        action_index[1] = 0
                    params[3] = (action_index[1] + 1) * 0.5 * np.pi
                    print('action: {} {}, discrete action: {}, params: {}'.format(index_action_mapping[action[0]],
                                                                                  action[1], action_index[1] + 1,
                                                                                  params))
            else:
                if action[0] in [1, ]:
                    params = (np.array(action_index[1:]) / self.multi_discrete_grid_size / 20).tolist()
                    # ([0,1] -> [0.000, 0.050]
                    # params = params.append(skill_object_offset_params[action[0]][action[1]][3])
        elif self.action_space_type == 'continuous':
            # action = self.action_list[int(action_index[0] * (int(self.num_discrete_action / 2) - 0.1) + (self.num_discrete_action - 1) / 2)]
            action_i = int(np.argmax(action_index[0:self.num_discrete_action]))
            print('action_i: ', action_i)
            action = self.action_list[action_i]
            params = skill_object_offset_params[action[0]][action[1]]
            if self.config['task'] in ['putting_away_Halloween_decorations']:
                if action[0] in [4, 5]:
                    new_params = action_index[self.num_discrete_action:]
                    params[1] = new_params[1] * 0.175 - 0.675  # [-0.5, -0.85]
                if action[0] in [2, ]:
                    new_params = action_index[self.num_discrete_action:]
                    params[0] = new_params[1] * 0.25 + 0.45  # [0.2, 0.7]
            elif self.config['task'] in ['throwing_away_leftovers']:
                # if action[0] in [1, ]:
                #     params = action_index[self.num_discrete_action:]
                #     params[0] = params[0] * 0.03
                #     params[1] = params[1] * 0.03
                #     params[2] = params[2] * 0.005 + 0.025
                # elif action[0] in [0, ] and action[1] == 'countertop.n.01_1':
                if action[0] in [0, ] and action[1] == 'countertop.n.01_1':
                    new_params = action_index[self.num_discrete_action:]
                    # print('params, new_params: ', params, new_params)
                    params[3] = new_params[3] * 0.4 * np.pi + 0.5 * np.pi
            print('action: {} {}, params: {}'.format(index_action_mapping[action[0]], action[1], params))
        else:  # 'discrete'
            action = copy.deepcopy(self.action_list[action_index])
            # print(self.action_list)
            if self.config['task'] in ['cleaning_microwave_oven'] and not(action[1] in ['towel']):
                # print(action)
                params = skill_object_offset_params[action[0]][action[1] + '-' + self.config['task']]
            else:
                params = skill_object_offset_params[action[0]][action[1]]

            if len(action) == 3:
                params = params[action[2]]
                # print('action 3, params: ', params)
            # if action_index == 0:
            #     params[3] = params[3] - 0.4 * np.pi
            # elif action_index == 8:
            #     params[3] = params[3] + 0.3 * np.pi
        self.env.current_step += 1
        self.current_step = self.env.current_step

        hit_normal = (0.0, 0.0, 1.0)  # default hit normal
        # print('cabinet pos: ', self.task_obj_list['cabinet.n.01_1'].states[Pose].get_value()[0])
        if action[0] == 0:  # base move to
            skip_move_flag = False
            if self.obj_pose_check:
                if self.config['task'] in ['putting_away_Halloween_decorations']:
                    obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
                    # print(np.array_equal(obj_pos, np.array([0, 0, 0])))
                    if action[1] in ['pumpkin.n.02_1', 'pumpkin.n.02_2']:
                        if action[1] not in self.initial_pos_dict:
                            self.initial_pos_dict[action[1]] = obj_pos
                        else:
                            print(action[1], obj_pos, self.initial_pos_dict[action[1]])
                            print('np.abs(np.sum(self.initial_pos_dict[action[1]] - obj_pos)): ',
                                  np.abs(np.sum(self.initial_pos_dict[action[1]] - obj_pos)))
                            if np.abs(np.sum(self.initial_pos_dict[action[1]] - obj_pos)) > 1e-1:
                                skip_move_flag = True
                                if self.print_log:
                                    print('move {}, ignored'.format(action[1]))
            if not skip_move_flag:
                obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
                obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]

                # process the offset from object frame to world frame
                mat = quat2mat(obj_rot_XYZW)
                vector = mat @ np.array(params[:3])

                # acquire the base direction
                euler = mat2euler(mat)
                target_yaw = euler[-1] + params[3]

                plan = self.planner.plan_base_motion([obj_pos[0] + vector[0], obj_pos[1] + vector[1], target_yaw])
                # print('plan: ', plan, 'obj_pos', obj_pos, 'obj_rot_XYZW', obj_rot_XYZW, )
                if self.visualize_arm_path_flag:
                    if plan is not None and len(plan) > 0:
                        # self.planner.dry_run_base_plan(plan)
                        self.planner.visualize_base_path(plan)

                if self.print_log:
                    print('move {}'.format(action[1]))

        elif action[0] == 1:  # arm pick
            if action[1] in ['towel'] and self.config['task'] in ['cleaning_microwave_oven']:
                # towel: 132
                # plt.imshow(self.state['ins_seg'])
                # plt.show()
                # print('self.env.scene.object_states: ', self.env.scene.object_states.keys())
                pos, orn = p.getBasePositionAndOrientation(132)  # towel: 132
                # print('pos, orn: ', pos, orn)
                # action[1] = action[1].split('@')[-1]
                # obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
                # obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]
                # print('obj_pos, obj_rot_XYZW: ', obj_pos, obj_rot_XYZW)
                obj_pos = list(pos)
                obj_rot_XYZW = list(orn)
            else:
                obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
                obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]

            # process the offset from object frame to world frame
            mat = quat2mat(obj_rot_XYZW)
            vector = mat @ np.array(params[:3])

            pick_place_pos = copy.deepcopy(obj_pos)
            pick_place_pos[0] += vector[0]
            pick_place_pos[1] += vector[1]
            pick_place_pos[2] += vector[2]

            pre_interaction_path, approaching_path, interaction_path = None, None, None
            pre_interaction_path, interaction_path = self.planner.plan_ee_pick(pick_place_pos, pre_grasping_distance=0.1)
            if self.visualize_arm_path_flag:
                if interaction_path is not None and len(interaction_path) != 0:
                    print("Visualizing pick")
                    self.planner.visualize_arm_path(interaction_path)
                    self.planner.visualize_arm_path(interaction_path, reverse_path=True)
                    print("End of the pick visualization")
                else:
                    logging.error("MP couldn't find path to pick.")
            # self.planner.execute_arm_pick(plan, pick_place_pos, -np.array(hit_normal))
            # print('plan: ', plan)
            if self.print_log:
                print('pick {}'.format(action[1], ))

        elif action[0] == 2:  # arm place
            obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
            obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]

            # process the offset from object frame to world frame
            mat = quat2mat(obj_rot_XYZW)
            np_array = np.array(params[:3])
            np_array[0] += random.uniform(0, 0.2)
            vector = mat @ np_array

            pick_place_pos = copy.deepcopy(obj_pos)
            pick_place_pos[0] += vector[0]
            pick_place_pos[1] += vector[1]
            pick_place_pos[2] += vector[2]

            pre_interaction_path, _ = self.planner.plan_ee_drop(pick_place_pos, dropping_distance=0.1, obj_name=action[1]+'-'+self.config['task'])
            # print('pre_interaction_path: ', pre_interaction_path)
            # self.planner.execute_arm_place(plan, pick_place_pos, -np.array(hit_normal))
            if self.visualize_arm_path_flag:
                if pre_interaction_path is not None and len(pre_interaction_path) != 0:
                    print("Visualizing drop")
                    self.planner.visualize_arm_path(pre_interaction_path)
                    self.planner.visualize_arm_path(interaction_path, reverse_path=True)
                    print("End of the drop visualization")
                else:
                    logging.error("MP couldn't find path to drop.")

            if self.print_log:
                print('place {}, pos: {}'.format(action[1], pick_place_pos))

        elif action[0] == 3:  # arm toggle
            obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
            obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]

            # process the offset from object frame to world frame
            mat = quat2mat(obj_rot_XYZW)
            vector = mat @ np.array(params[:3])

            toggle_pos = copy.deepcopy(obj_pos)
            toggle_pos[0] += vector[0]
            toggle_pos[1] += vector[1]
            toggle_pos[2] += vector[2]

            pre_interaction_path, interaction_path = self.planner.plan_ee_toggle(toggle_pos, -np.array(hit_normal), pre_toggling_distance=0.1,)
            if self.visualize_arm_path_flag:
                if interaction_path is not None and len(interaction_path) != 0:
                    print("Visualizing toggle")
                    self.planner.visualize_arm_path(interaction_path)
                    self.planner.visualize_arm_path(interaction_path, reverse_path=True)
                    print("End of the toggle visualization")
                else:
                    logging.error("MP couldn't find path to toggle.")
            # self.planner.execute_arm_toggle(plan, toggle_pos, -np.array(hit_normal))
            # print('toggle plan: ', plan)
            if self.print_log:
                print('toggle {}'.format(action[1]))

        elif action[0] == 4:  # arm pull
            obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
            obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]

            # process the offset from object frame to world frame
            mat = quat2mat(obj_rot_XYZW)
            vector = mat @ np.array(params[:3])

            pick_place_pos = copy.deepcopy(obj_pos)
            pick_place_pos[0] += vector[0]
            pick_place_pos[1] += vector[1]
            pick_place_pos[2] += vector[2]

            # plan = self.planner.plan_ee_pull(pick_place_pos, hit_normal=np.array((-1.0, 0.0, 0.0)))
            # self.planner.execute_arm_pull(plan, pick_place_pos, -np.array(hit_normal))

            pre_interaction_path, approach_interaction_path, interaction_path = self.planner.plan_ee_pull(pick_place_pos, np.array((-1.0, 0.0, 0.0)))
            if self.visualize_arm_path_flag:
                if interaction_path is not None and len(interaction_path) != 0:
                    print("Visualizing toggle")
                    self.planner.visualize_arm_path(interaction_path)
                    self.planner.visualize_arm_path(interaction_path, reverse_path=True)
                    print("End of the toggle visualization")
                else:
                    logging.error("MP couldn't find path to toggle.")

            if self.print_log:
                print('pull {}'.format(action[1]))

        elif action[0] == 5:  # arm push
            obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
            obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]

            # process the offset from object frame to world frame
            mat = quat2mat(obj_rot_XYZW)
            vector = mat @ np.array(params[:3])

            pick_place_pos = copy.deepcopy(obj_pos)
            pick_place_pos[0] += vector[0]
            pick_place_pos[1] += vector[1]
            pick_place_pos[2] += vector[2]

            pre_interaction_path, interaction_path = self.planner.plan_ee_push(pick_place_pos, np.array((1.0, 0.0, 0.0)))
            # self.planner.execute_arm_push(plan, pick_place_pos, np.array((1.0, 0.0, 0.0)))
            if self.visualize_arm_path_flag:
                if interaction_path is not None and len(interaction_path) != 0:
                    print("Visualizing push")
                    self.planner.visualize_arm_path(interaction_path)
                    self.planner.visualize_arm_path(interaction_path, reverse_path=True)
                    print("End of the push visualization")
                else:
                    logging.error("MP couldn't find path to push.")
            if self.print_log:
                print('push {}'.format(action[1]))

        elif action[0] == 6:  # vis pick
            # print(6, 'self.task_obj_list[action[1]].category,: ', self.task_obj_list[action[1]].category,)  # hamburger
            ins_seg = np.round(self.state['ins_seg'][:, :, 0]).astype(int)
            # print('max(ins_seg), min(ins_seg): ', np.max(ins_seg), np.min(ins_seg))
            id_ins_seg = self.env.simulator.renderer.get_pb_ids_for_instance_ids(ins_seg)
            obj_id = self.task_obj_list[action[1]].get_body_ids()
            # print('action[0]: {}, action[1]: {}, obj_id: {}'.format(action[0], action[1], obj_id))
            if self.print_log:
                # plt.imshow(self.state['ins_seg'])
                # plt.show()
                pass
            if obj_id in id_ins_seg:
                skip_move_flag = False
                if self.obj_pose_check:
                    if self.config['task'] in ['putting_away_Halloween_decorations']:
                        obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
                        # print(np.array_equal(obj_pos, np.array([0, 0, 0])))
                        if action[1] in ['pumpkin.n.02_1', 'pumpkin.n.02_2']:
                            if action[1] not in self.initial_pos_dict:
                                self.initial_pos_dict[action[1]] = obj_pos
                            else:
                                print(action[1], obj_pos, self.initial_pos_dict[action[1]])
                                print('np.abs(np.sum(self.initial_pos_dict[action[1]] - obj_pos)): ',
                                      np.abs(np.sum(self.initial_pos_dict[action[1]] - obj_pos)))
                                if np.abs(np.sum(self.initial_pos_dict[action[1]] - obj_pos)) > 1e-1:
                                    skip_move_flag = True
                                    if self.print_log:
                                        print('vis pick move {}, ignored'.format(action[1]))
                if not skip_move_flag:
                    move_params = copy.deepcopy(params[:4])
                    obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
                    obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]

                    # process the offset from object frame to world frame
                    mat = quat2mat(obj_rot_XYZW)
                    vector = mat @ np.array(move_params[:3])

                    # acquire the base direction
                    euler = mat2euler(mat)
                    target_yaw = euler[-1] + move_params[3]

                    plan = self.planner.plan_base_motion([obj_pos[0] + vector[0], obj_pos[1] + vector[1], target_yaw])
                    if plan is not None and len(plan) > 0:
                        # self.planner.dry_run_base_plan(plan)
                        self.planner.visualize_base_path(plan)

                    if self.print_log:
                        print('vis pick move {}'.format(action[1]))

                pick_params = copy.deepcopy(params[4:])
                obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
                obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]

                # process the offset from object frame to world frame
                mat = quat2mat(obj_rot_XYZW)
                vector = mat @ np.array(pick_params[:3])

                pick_place_pos = copy.deepcopy(obj_pos)
                pick_place_pos[0] += vector[0]
                pick_place_pos[1] += vector[1]
                pick_place_pos[2] += vector[2]
                # print('pick_pos: ', pick_place_pos)
                pre_interaction_path, approaching_path, interaction_path = None, None, None
                pre_interaction_path, interaction_path = self.planner.plan_ee_pick(pick_place_pos, pre_grasping_distance=0.1)
                if self.visualize_arm_path_flag:
                    if interaction_path is not None and len(interaction_path) != 0:
                        print("Visualizing pick")
                        self.planner.visualize_arm_path(interaction_path)
                        self.planner.visualize_arm_path(interaction_path, reverse_path=True)
                        print("End of the pick visualization")
                    else:
                        logging.error("MP couldn't find path to pick.")
                # print('plan: ', plan)
                if self.print_log:
                    print('vis_pick {}'.format(action[1], ))
                # print('pick params: ', params)
            else:
                if self.print_log:
                    print('vis_pick {} not done'.format(action[1], ))

        elif action[0] == 7:  # vis place
            # print(7, 'self.task_obj_list[action[1]].category,: ', self.task_obj_list[action[1]].category, )
            ins_seg = np.round(self.state['ins_seg'][:, :, 0]).astype(int)
            # print('max(ins_seg), min(ins_seg): ', np.max(ins_seg), np.min(ins_seg))
            id_ins_seg = self.env.simulator.renderer.get_pb_ids_for_instance_ids(ins_seg)
            obj_id = self.task_obj_list[action[1]].get_body_ids()
            # print('action[0]: {}, action[1]: {}, obj_id: {}'.format(action[0], action[1], obj_id))
            if self.print_log:
                # plt.imshow(self.state['ins_seg'])
                # plt.show()
                pass
            # print('self.state.keys(): ', self.state.keys()) # self.state.keys():  odict_keys(['ins_seg', 'scan', 'occupancy_grid'])
            if obj_id in id_ins_seg:
                skip_move_flag = False
                if self.obj_pose_check:
                    if self.config['task'] in ['putting_away_Halloween_decorations']:
                        obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
                        # print(np.array_equal(obj_pos, np.array([0, 0, 0])))
                        if action[1] in ['pumpkin.n.02_1', 'pumpkin.n.02_2']:
                            if action[1] not in self.initial_pos_dict:
                                self.initial_pos_dict[action[1]] = obj_pos
                            else:
                                print(action[1], obj_pos, self.initial_pos_dict[action[1]])
                                print('np.abs(np.sum(self.initial_pos_dict[action[1]] - obj_pos)): ',
                                      np.abs(np.sum(self.initial_pos_dict[action[1]] - obj_pos)))
                                if np.abs(np.sum(self.initial_pos_dict[action[1]] - obj_pos)) > 1e-1:
                                    skip_move_flag = True
                                    if self.print_log:
                                        print('move {}, ignored'.format(action[1]))
                if not skip_move_flag:
                    move_params = copy.deepcopy(params[:4])
                    obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
                    obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]

                    # process the offset from object frame to world frame
                    mat = quat2mat(obj_rot_XYZW)
                    vector = mat @ np.array(move_params[:3])

                    # acquire the base direction
                    euler = mat2euler(mat)
                    target_yaw = euler[-1] + move_params[3]

                    plan = self.planner.plan_base_motion([obj_pos[0] + vector[0], obj_pos[1] + vector[1], target_yaw])
                    if self.visualize_arm_path_flag:
                        if plan is not None and len(plan) > 0:
                            # self.planner.dry_run_base_plan(plan)
                            self.planner.visualize_base_path(plan)

                    if self.print_log:
                        print('vis place move {}'.format(action[1]))

                pick_params = copy.deepcopy(params[4:])
                obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
                obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]

                # process the offset from object frame to world frame
                mat = quat2mat(obj_rot_XYZW)
                vector = mat @ np.array(pick_params[:3])

                pick_place_pos = copy.deepcopy(obj_pos)
                pick_place_pos[0] += vector[0]
                pick_place_pos[1] += vector[1]
                pick_place_pos[2] += vector[2]
                # print('place_pos: ', pick_place_pos)
                pre_interaction_path, _ = self.planner.plan_ee_drop(pick_place_pos)
                # self.planner.execute_arm_place(plan, pick_place_pos, -np.array(hit_normal))
                if pre_interaction_path is not None and len(pre_interaction_path) != 0:
                    print("Visualizing drop")
                    self.planner.visualize_arm_path(pre_interaction_path)
                    self.planner.visualize_arm_path(interaction_path, reverse_path=True)
                    print("End of the drop visualization")
                else:
                    logging.error("MP couldn't find path to drop.")

                if self.print_log:
                    print('vis_place {}'.format(action[1]))
                # print('place params: ', params)
            else:
                if self.print_log:
                    print('vis_place {} not done'.format(action[1]))

        elif action[0] == 8:  # vis pull
            # print(6, 'self.task_obj_list[action[1]].category,: ', self.task_obj_list[action[1]].category,)  # hamburger
            ins_seg = np.round(self.state['ins_seg'][:, :, 0]).astype(int)
            # print('max(ins_seg), min(ins_seg): ', np.max(ins_seg), np.min(ins_seg))
            id_ins_seg = self.env.simulator.renderer.get_pb_ids_for_instance_ids(ins_seg)
            obj_id = self.task_obj_list[action[1]].get_body_ids()
            # print('action[0]: {}, action[1]: {}, obj_id: {}'.format(action[0], action[1], obj_id))
            if self.print_log:
                # plt.imshow(self.state['ins_seg'])
                # plt.show()
                pass
            if obj_id in id_ins_seg:
                obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
                obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]

                # process the offset from object frame to world frame
                mat = quat2mat(obj_rot_XYZW)
                vector = mat @ np.array(params[:3])

                pick_place_pos = copy.deepcopy(obj_pos)
                pick_place_pos[0] += vector[0]
                pick_place_pos[1] += vector[1]
                pick_place_pos[2] += vector[2]

                # plan = self.planner.plan_ee_pull(pick_place_pos, hit_normal=np.array((-1.0, 0.0, 0.0)))
                # self.planner.execute_arm_pull(plan, pick_place_pos, -np.array(hit_normal))

                pre_interaction_path, approach_interaction_path, interaction_path = self.planner.plan_ee_pull(pick_place_pos, np.array((-1.0, 0.0, 0.0)))
                # self.planner.execute_arm_pull(plan, pick_place_pos, np.array((-1.0, 0.0, 0.0)))
                if self.visualize_arm_path_flag:
                    if pre_interaction_path is not None and len(pre_interaction_path) != 0:
                        print("Visualizing pull")
                        self.planner.visualize_arm_path(pre_interaction_path)
                        self.planner.visualize_arm_path(interaction_path, reverse_path=True)
                        print("End of the pull visualization")
                    else:
                        logging.error("MP couldn't find path to pull.")

                if self.print_log:
                    print('vis pull {}'.format(action[1]))
            else:
                if self.print_log:
                    print('vis pull {} not done'.format(action[1]))

        elif action[0] == 9:  # vis push
            # print(6, 'self.task_obj_list[action[1]].category,: ', self.task_obj_list[action[1]].category,)  # hamburger
            ins_seg = np.round(self.state['ins_seg'][:, :, 0]).astype(int)
            # print('max(ins_seg), min(ins_seg): ', np.max(ins_seg), np.min(ins_seg))
            id_ins_seg = self.env.simulator.renderer.get_pb_ids_for_instance_ids(ins_seg)
            obj_id = self.task_obj_list[action[1]].get_body_ids()
            # print('action[0]: {}, action[1]: {}, obj_id: {}'.format(action[0], action[1], obj_id))
            if self.print_log:
                # plt.imshow(self.state['ins_seg'])
                # plt.show()
                pass
            if obj_id in id_ins_seg:
                obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
                obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]

                # process the offset from object frame to world frame
                mat = quat2mat(obj_rot_XYZW)
                vector = mat @ np.array(params[:3])

                pick_place_pos = copy.deepcopy(obj_pos)
                pick_place_pos[0] += vector[0]
                pick_place_pos[1] += vector[1]
                pick_place_pos[2] += vector[2]

                pre_interaction_path, interaction_path = self.planner.plan_ee_push(pick_place_pos, np.array((1.0, 0.0, 0.0)))
                # self.planner.execute_arm_push(plan, pick_place_pos, np.array((1.0, 0.0, 0.0)))
                if self.visualize_arm_path_flag:
                    if interaction_path is not None and len(interaction_path) != 0:
                        print("Visualizing push")
                        self.planner.visualize_arm_path(interaction_path)
                        self.planner.visualize_arm_path(interaction_path, reverse_path=True)
                        print("End of the push visualization")
                    else:
                        logging.error("MP couldn't find path to push.")

                if self.print_log:
                    print('vis push {}'.format(action[1]))
            else:
                if self.print_log:
                    print('vis push {} not done'.format(action[1]))

        self.env.simulator.sync()

        self.state = self.env.get_state()
        if 'state_vec' in self.config["output"]:
            print(self.state.keys())
            # new_state = {}
            # new_state['occupancy_grid'] = self.state['occupancy_grid']
            robot = self.env.robots[0]
            is_grasping = (robot.is_grasping() == True)
            robot_position_xy = robot.get_position()[:2]
            self.state['state_vec'] = np.concatenate((np.array([robot_position_xy]), np.array([[is_grasping]]).astype(np.float32)), axis=1)
        reward, info = self.env.task.get_reward(self.env)
        if self.config['task'] in ['cleaning_microwave_oven']:
            if self.soak_bonus and self.env.scene.objects_by_id[132].states[Soaked].get_value():
                reward = reward + 0.5
                self.soak_bonus = False

        self.accum_reward = self.accum_reward + reward
        if self.dense_reward:
            if self.config['task'] in ['throwing_away_leftovers', ]:
                reward = reward - 0.01
            elif self.config['task'] == 'installing_a_printer':
                reward = reward - 0.1
            else:  # in ['putting_away_Halloween_decorations']:
                reward = reward - 0.01

        done, info = self.env.task.get_termination(self.env)
        # if self.print_log:
        print('reward: ', reward)
        # print(self.step_counter, 'info: ', info)
        # info:  {'done': False, 'success': False, 'goal_status': {'satisfied': [], 'unsatisfied': [0]}}
        if done and self.env.automatic_reset:
            info["last_observation"] = self.state
            self.state = self.env.reset()

        # https://github.com/DLR-RM/stable-baselines3/blob/master/docs/common/logger.rst#eval
        # if done and self.step_counter <= self.config['max_step']:
        #     info["is_success"] = True
        #     if info['is_success'] != info['success']:
        #         print('is_success: {}, success: {}'.format(info['is_success'], info['success']))
        # elif self.step_counter == self.config['max_step'] and not done:
        #     info["is_success"] = False
        #     if info['is_success'] != info['success']:
        #         print('is_success: {}, success: {}'.format(info['is_success'], info['success']))
        if done:
            info["is_success"] = info["success"]
            if self.is_success_count:
                self.is_success_list.append(info['is_success'])
            print('is_success: {}'.format(info['is_success']))
        self.state['accum_reward'] = self.accum_reward
        print('self.accum_reward: ', self.state['accum_reward'])
        # print('info: ', info)
        if self.obj_joint_obs:
            self.obj_joint = self.get_joint()
            self.state['obj_joint'] = self.obj_joint
        return self.state, reward, done, info

    def reset(self):
        """
        Reset episode.
        """
        self.step_counter = 0
        self.state = self.env.reset()
        self.accum_reward = np.array([0.])
        if self.config['task'] in ['putting_away_Halloween_decorations']:
            # mode='zero'. Change here https://github.com/StanfordVL/iGibson/blob/master/igibson/scenes/igibson_indoor_scene.py#L795
            # from `p.resetJointState(body_id, joint_id, 0.0)` to `p.resetJointState(body_id, joint_id, 0.2)`
            self.env.scene.open_all_objs_by_category(category='bottom_cabinet', mode='zero')
            # self.env.scene.open_all_objs_by_category(category='bottom_cabinet', mode='max')
            print('bottom_cabinet opened!')
            self.initial_pos_dict = {}
            self.env.simulator.sync()
            # self.env.scene.open_all_objs_by_category(category='top_cabinet', mode='max')
            # print('top_cabinet opened!')
        elif self.config['task'] in ['cleaning_microwave_oven']:
            self.env.scene.open_all_objs_by_category(category='microwave', mode='max')
            print('microwave opened!')
            self.initial_pos_dict = {}
            self.env.simulator.sync()
        print("new trial!!!, success rate: {}".format(np.mean(self.is_success_list)))
        self.state['accum_reward'] = self.accum_reward
        self.state['obj_joint'] = self.obj_joint
        self.soak_bonus = True
        return self.state

    def close(self):
        """
        Reset episode.
        """
        self.env.close()

    def get_joint(self, id=None, mode='zero'):
        # body_joint_pairs = []
        # all_joint_info = p.getJointInfo('pumpkin.n.02_1', joint_id)
        for obj in self.env.scene.objects_by_category['bottom_cabinet']:
            print('obj: ', obj)
            for body_id in obj.get_body_ids():
                print('body_id: ', body_id)
                for joint_id in range(p.getNumJoints(body_id)):
                    print('joint_id: ', joint_id)
                    # cache current physics state
                    # state_id = p.saveState()
                    all_joint_info = p.getJointInfo(body_id, joint_id)
                    print('all_joint_info: {}, all_joint_info.shape: {}'
                          .format(all_joint_info, len(all_joint_info)))
                    # j_low, j_high = p.getJointInfo(body_id, joint_id)[8:10]
                    # j_type = p.getJointInfo(body_id, joint_id)[2]
                    # parent_idx = p.getJointInfo(body_id, joint_id)[-1]
        # print('body_joint_pairs: ', body_joint_pairs)
        return all_joint_info

    def set_joints(self, id=None, relative_joints_pos=1.0):
        body_joint_pairs = []
        for obj in self.env.scene.objects_by_category['bottom_cabinet']:
            print('obj: ', obj)
            for body_id in obj.get_body_ids():
                for joint_id in range(p.getNumJoints(body_id)):
                    # cache current physics state
                    state_id = p.saveState()
                    all_joint_info = p.getJointInfo(body_id, joint_id)
                    print('all_joint_info: {}, all_joint_info.shape: {}'
                          .format(all_joint_info, all_joint_info.shape))  # , 17
                    j_low, j_high = p.getJointInfo(body_id, joint_id)[8:10]
                    j_type = p.getJointInfo(body_id, joint_id)[2]
                    parent_idx = p.getJointInfo(body_id, joint_id)[-1]
                    if j_type not in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                        p.removeState(state_id)
                        continue
                    # this is the continuous joint
                    if j_low >= j_high:
                        p.removeState(state_id)
                        continue
                    # this is the 2nd degree joint, ignore for now
                    if parent_idx != -1:
                        p.removeState(state_id)
                        continue

                    # if mode == "max":
                    #     # try to set the joint to the maxr value until no collision
                    #     # step_size is 5cm for prismatic joint and 5 degrees for revolute joint
                    #     step_size = np.pi / 36.0 if j_type == p.JOINT_REVOLUTE else 0.05
                    #     for j_pos in np.arange(0.0, j_high + step_size, step=step_size):
                    #         p.resetJointState(body_id, joint_id, j_high - j_pos)
                    #         p.stepSimulation()
                    #         has_collision = self.check_collision(body_a=body_id, link_a=joint_id)
                    #         restoreState(state_id)
                    #         if not has_collision:
                    #             p.resetJointState(body_id, joint_id, j_high - j_pos)
                    #             break
                    p.resetJointState(body_id, joint_id, relative_joints_pos)

                    body_joint_pairs.append((body_id, joint_id))
                    # Remove cached state to avoid memory leak.
                    # p.removeState(state_id)
        print('body_joint_pairs: ', body_joint_pairs)


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
    # np.random.seed(0)
    # random.seed(0)

    env = SkillEnv(config_file=args.config, mode=args.mode)

    step_time_list = []
    # action_list = action_dict['throwing_away_leftovers']
    # action_list = action_dict['putting_away_Halloween_decorations']
    action_list = action_dict['cleaning_microwave_oven']
    for episode in range(1):
        print("Episode: {}".format(episode))
        start = time.time()
        env.reset()
        # for i in range(len(action_list_throwing_away_leftovers)):  # 10 seconds

        for i in range(len(action_list)):  # 10 seconds
            # state, reward, done, info = env.step([i, 0, 0, 5])
            # state, reward, done, info = env.step([i, 0, 0, 0.25, 0.5*np.pi])
            state, reward, done, info = env.step(i)
            print("{}, reward: {}, done: {}, is_success: {}".format(i, reward, done, info['success']))
            # if done:
            #     break
        print("Episode finished after {} timesteps, took {} seconds.".format(env.current_step, time.time() - start))
    env.close()
