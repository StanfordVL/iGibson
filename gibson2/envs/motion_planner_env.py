# from gibson2.external.pybullet_tools.utils import set_base_values, joint_from_name, set_joint_position, \
#     set_joint_positions, add_data_path, connect, plan_base_motion, plan_joint_motion, enable_gravity, \
#     joint_controller, dump_body, load_model, joints_from_names, user_input, disconnect, get_joint_positions, \
#     get_link_pose, link_from_name, HideOutput, get_pose, wait_for_user, dump_world, plan_nonholonomic_motion, \
#     set_point, create_box, stable_z, control_joints, get_max_limits, get_min_limits, get_base_values, \
#     plan_base_motion_2d, get_sample_fn, add_p2p_constraint, remove_constraint, set_base_values_with_z

from gibson2.external.pybullet_tools.utils import control_joints
from gibson2.external.pybullet_tools.utils import get_joint_positions
from gibson2.external.pybullet_tools.utils import get_joint_velocities
from gibson2.external.pybullet_tools.utils import get_max_limits
from gibson2.external.pybullet_tools.utils import get_min_limits
from gibson2.external.pybullet_tools.utils import plan_joint_motion
from gibson2.external.pybullet_tools.utils import link_from_name
from gibson2.external.pybullet_tools.utils import set_joint_positions
from gibson2.external.pybullet_tools.utils import get_sample_fn
from gibson2.external.pybullet_tools.utils import set_base_values_with_z
from gibson2.external.pybullet_tools.utils import get_base_values
from gibson2.external.pybullet_tools.utils import plan_base_motion_2d
from gibson2.external.pybullet_tools.utils import joints_from_names
from gibson2.utils.utils import l2_distance, quatToXYZW
from gibson2.utils.utils import rotate_vector_2d, rotate_vector_3d
from gibson2.core.physics.interactive_objects import BoxShape
from gibson2.core.physics.interactive_objects import YCBObject
from gibson2.core.physics.interactive_objects import InteractiveObj
from gibson2.core.physics.interactive_objects import VisualMarker
from gibson2.envs.locomotor_env import NavigateRandomEnv
from gibson2.core.render.utils import quat_pos_to_mat
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2mat
from IPython import embed

import gym
import numpy as np
import os
import pybullet as p
import cv2
from PIL import Image
import time
import argparse
import gibson2
import logging
import string
import random
import collections
import copy


class MotionPlanningBaseArmEnv(NavigateRandomEnv):
    def __init__(self,
                 config_file,
                 model_id=None,
                 collision_reward_weight=0.0,
                 mode='headless',
                 action_timestep=1 / 10.0,
                 physics_timestep=1 / 240.0,
                 device_idx=0,
                 random_height=False,
                 automatic_reset=False,
                 arena=None,
                 render_to_tensor=False,
                 action_map=False,
                 channel_first=False,
                 draw_path_on_map=False,
                 draw_objs_on_map=False,
                 rotate_occ_grid=False,
                 randomize_object_pose=True,
                 log_dir=None,
                 fine_motion_plan=None,
                 base_mp_algo='birrt',
                 arm_mp_algo='birrt',
                 ):
        super(MotionPlanningBaseArmEnv, self).__init__(
            config_file,
            model_id=model_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            automatic_reset=automatic_reset,
            random_height=random_height,
            device_idx=device_idx,
            render_to_tensor=render_to_tensor)

        self.arena = arena
        print('arena:', self.arena)
        self.new_potential = None
        self.collision_reward_weight = collision_reward_weight
        self.action_map = action_map
        self.channel_first = channel_first
        self.randomize_object_pose = randomize_object_pose
        self.log_dir = log_dir
        self.base_mp_algo = base_mp_algo
        self.base_mp_resolutions = np.array([0.05, 0.05, 0.05])
        self.arm_mp_algo = arm_mp_algo

        # draw the shortest path on the occupancy map
        self.draw_path_on_map = draw_path_on_map
        if self.draw_path_on_map:
            assert self.arena not in ['push_chairs', 'push_drawers']

        # draw objects of interest (chairs, drawers, etc) on the occupancy map
        self.draw_objs_on_map = draw_objs_on_map
        if self.draw_objs_on_map:
            assert self.arena in ['push_chairs', 'push_drawers']

        # disable navigation success
        if self.arena in ['push_chairs', 'push_drawers',
                          'tabletop_manip', 'tabletop_reaching']:
            self.dist_tol = -1.0

        # tabletop_manip and tabletop_reaching has shorter episode length and
        # larger head tilt angle
        head_tilt_angle = quat2euler(
            p.getJointInfo(self.robots[0].robot_ids[0], 5)[15])[1]
        if self.arena in ['tabletop_manip', 'tabletop_reaching']:
            self.max_step = int(self.max_step * 0.4)
            assert np.abs(head_tilt_angle - np.deg2rad(45)) < 1e-3, \
                'head tilte angle should be 45 degrees for {}'.format(
                    self.arena)
        else:
            assert np.abs(head_tilt_angle - np.deg2rad(10)) < 1e-3, \
                'head tilte angle should be 10 degrees for {}'.format(
                    self.arena)

        self.rotate_occ_grid = rotate_occ_grid
        self.fine_motion_plan = self.config.get('fine_motion_plan', True)
        if fine_motion_plan is not None:
            self.fine_motion_plan = fine_motion_plan
        if self.arena in ['tabletop_manip', 'tabletop_reaching'] and \
                not self.fine_motion_plan:
            print('WARNING: tabletop requires fine motion planning')
        self.arm_subgoal_threshold = 0.05
        self.failed_subgoal_penalty = -0.0
        self.arm_interaction_length = 0.25
        self.continuous_action = False
        self.base_subgoal_success = False
        self.arm_subgoal_success = False

        self.prepare_motion_planner()
        self.update_action_space()
        self.update_observation_space()
        self.update_visualization()
        self.prepare_scene()
        self.prepare_mp_obstacles()
        self.prepare_logging()
        # np.random.seed(0)

    def prepare_logging(self):
        if self.log_dir is not None:
            logger = logging.getLogger('log')
            logger.setLevel(logging.INFO)

            filename = 'envlog-'+''.join(random.choices(
                string.ascii_uppercase + string.digits, k=8))
            logpath = os.path.join(self.log_dir, filename + '.log')

            ch = logging.FileHandler(logpath)
            ch.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(ch)

            self.logger = logger
            self.logger.info('model_id: ' + self.scene.model_id)
            self.logger.info('arena: ' + self.arena)
            self.logger.info('fine_motion_plan: ' + str(self.fine_motion_plan))
            self.logger.info('base_mp_algo: ' + self.base_mp_algo)
            self.logger.info('arm_mp_algo: ' + self.arm_mp_algo)

        self.metric_keys = [
            'episode_return',
            'episode_length',
            'collision_step',
            'path_length',
            'geodesic_dist',
            'success',
            'spl',
            'dist_to_goal',
            'doors_opened',
            'drawers_closed_5',
            'drawers_closed_10',
            'chairs_pushed_5',
            'chairs_pushed_10',
            'base_mp_time',
            'base_mp_success',
            'base_mp_failure',
            'base_mp_num_waypoints',
            'base_mp_path_length',
            'arm_ik_time',
            'arm_ik_failure',
            'arm_mp_time',
            'arm_mp_success',
            'arm_mp_failure',
            'arm_mp_num_waypoints',
            'arm_mp_path_length',
        ]
        self.metrics = {
            key: collections.deque(maxlen=100) for key in self.metric_keys
        }
        self.episode_metrics = {
            key: 0.0 for key in self.metric_keys
        }

    def prepare_mp_obstacles(self):
        self.mp_obstacles_id = [self.mesh_id]
        if self.arena in ['push_door', 'button_door',
                          'random_manip', 'random_manip_atomic']:
            for obj in self.doors:
                self.mp_obstacles_id.append(obj.body_id)

        if self.arena == 'button_door':
            for obj in self.buttons:
                self.mp_obstacles_id.append(obj.body_id)

        if self.arena in ['obstacles', 'semantic_obstacles', 'push_chairs']:
            for obj in self.obstacles:
                self.mp_obstacles_id.append(obj.body_id)

        if self.arena == 'push_drawers':
            for obj in self.cabinet_drawers:
                self.mp_obstacles_id.append(obj.body_id)

        if self.arena in ['push_chairs',
                          'tabletop_manip', 'tabletop_reaching']:
            self.mp_obstacles_id.append(self.table.body_id)

        if self.arena == 'tabletop_manip':
            self.mp_obstacles_id.append(self.tabletop_object.body_id)

    def prepare_scene(self):
        self.doors = []
        self.walls = []
        self.obstacles = []
        self.buttons = []
        self.cabinet_drawers = []
        self.table = None
        self.tabletop_object = None

        if self.scene.model_id == 'Avonia':
            # push_door, button_door
            door_scales = [
                1.0,
                0.9,
            ]
            self.door_positions = [
                [-3.5, 0, 0.02],
                [-1.2, -2.47, 0.02],
            ]
            self.door_rotations = [np.pi / 2.0, -np.pi / 2.0]
            self.wall_poses = [
                [[-3.5, 0.47, 0.45],
                    quatToXYZW(euler2quat(0, 0, np.pi / 2.0), 'wxyz')],
                [[-3.5, -0.45, 0.45],
                    quatToXYZW(euler2quat(0, 0, -np.pi / 2.0), 'wxyz')],
            ]
            self.door_target_pos = [
                [[-5.5, -4.5], [-1.0, 1.0]],
                [[0.5, 2.0], [-4.5, -3.0]],
            ]

            # button_door
            button_scales = [
                2.0,
                2.0,
            ]
            self.button_positions = [
                [[-2.85, -2.85], [1.2, 1.7]],
                [[-2.1, -1.6], [-2.95, -2.95]]
            ]
            self.button_rotations = [
                -np.pi / 2.0,
                0.0,
            ]

            # obstacles
            self.obstacle_poses = [
                [[-3.5, 0.4, 0.7], [0.0, 0.0, 0.0, 1.0]],
                [[-3.5, -0.3, 0.7], [0.0, 0.0, 0.0, 1.0]],
                [[-3.5, -1.0, 0.7], [0.0, 0.0, 0.0, 1.0]],
            ]

            # semantic_obstacles
            self.semantic_obstacle_poses = [
                [[-3.5, 0.15, 0.63], [0.0, 0.0, 0.0, 1.0]],
                [[-3.5, -0.95, 0.63], [0.0, 0.0, 0.0, 1.0]],
            ]
            self.semantic_obstacle_masses = [
                1.0,
                10000.0,
            ]
            self.semantic_obstacle_colors = [
                [1.0, 0.0, 0.0, 1],
                [0.0, 1.0, 0.0, 1],
            ]

            self.initial_pos_range = np.array([[-1, 1.5], [-1, 1]])
            self.target_pos_range = np.array([[-5.5, -4.5], [-1.0, 1.0]])
            self.obstacle_dim = 0.35

        elif self.scene.model_id == 'gates_jan20':
            door_scales = [
                0.95,
                0.95,
            ]
            self.door_positions = [
                [-29.95, 0.7, 0.05],
                [-36, 0.7, 0.05],
            ]
            self.door_rotations = [np.pi, np.pi]
            self.wall_poses = []
            self.door_target_pos = [
                [[-30.5, -30], [-3, -1]],
                [[-36.75, -36.25], [-3, -1]],
            ]
            self.initial_pos_range = np.array([[-24, -22.5], [6, 9]])
            self.target_pos_range = np.array([[-40, -30], [1.25, 1.75]])

            # button_door
            button_scales = [
                2.0,
                2.0,
            ]
            self.button_positions = [
                [[-29.2, -29.2], [0.86, 0.86]],
                [[-35.2, -35.2], [0.86, 0.86]]
            ]
            self.button_rotations = [
                0.0,
                0.0,
            ]

            # semantic_obstacles
            self.semantic_obstacle_poses = [
                [[-28.1, 1.45, 0.62], [0.0, 0.0, 0.0, 1.0]],
            ]
            self.semantic_obstacle_masses = [
                1.0,
            ]
            self.semantic_obstacle_colors = [
                [1.0, 0.0, 0.0, 1],
            ]

            self.obstacle_dim = 0.25

        elif self.scene.model_id == 'candcenter':
            door_scales = [
                1.03
            ]
            self.door_positions = [
                [1.2, -2.15, 0],
            ]
            self.door_rotations = [np.pi]
            self.wall_poses = [
                [[1.5, -2.2, 0.25], quatToXYZW(euler2quat(0, 0, 0), 'wxyz')]
            ]
            self.door_target_pos = np.array([
                [[-1.0, 1.0], [-5.0, -3.0]]
            ])
            self.initial_pos_range = np.array([
                [5.0, 7.0], [-1.7, 0.3]
            ])
            self.initial_pos_range_near_door = np.array([
                [1.0, 1.4], [-1.6, -1.4]
            ])
            self.initial_orn_range_near_door = np.array([
                -np.pi / 2.0 - np.pi / 12.0, -np.pi / 2.0 + np.pi / 12.0,
            ])
            self.target_pos_range = np.array([
                [-3.75, -3.25], [-1, 0.0]
            ])
            # self.target_pos_range = np.array([
            #     [0.0, 0.0], [0.0, 0.0]
            # ])
            button_scales = [
                1.7,
            ]
            self.button_positions = [
                [[0.6, 0.6], [-2.1, -2.1]],
            ]
            self.button_rotations = [
                0.0,
            ]
            # semantic_obstacles
            self.semantic_obstacle_poses = [
                [[-2, -1.25, 0.61], [0.0, 0.0, 0.0, 1.0]],
                [[-2, -0.1, 0.61], [0.0, 0.0, 0.0, 1.0]],
            ]
            self.semantic_obstacle_masses = [
                1.0,
                10000.0,
            ]
            self.semantic_obstacle_colors = [
                [1.0, 0.0, 0.0, 1],
                [0.0, 1.0, 0.0, 1],
            ]
            self.obstacle_dim = 0.35

        elif self.scene.model_id == 'Samuels':
            # for push_drawers and push_chairs tasks
            self.initial_pos_range = np.array([
                [-4.2, -4.2], [-0.8, -0.8]
            ])
            self.target_pos_range = np.array([
                [-4.2, -4.2], [-0.8, -0.8]
            ])
            # self.chair_poses = [
            #     [[-5.2, 0.5, 0.63], [0, 0, 1, 1]],
            #     [[-3.4, 0.5, 0.63], [0, 0, -1, 1]],
            # ]
            self.chair_poses = [
                [[-5.2, 0.5, 0.462], np.pi * -0.71],
                [[-3.4, 0.5, 0.462], np.pi * 0.29],
            ]
            if self.arena == 'push_chairs':
                self.table_pose = [[-4.3, 0.5, 0.55], -np.pi / 2.0]
            elif self.arena == 'tabletop_manip':
                self.table_pose = [[-4.25, -0.15, 0.4], -np.pi / 2.0]
            elif self.arena == 'tabletop_reaching':
                self.table_pose = [[-4.25, 0.15, 0.4], -np.pi / 2.0]

            door_scales = [
                1.3,
                1.0,
                1.0
            ]
            self.chair_pose_transform = [
                [
                    0.0,
                    np.array([-0.1, 0.0, 0]),
                    np.array([0.1, 0.8, 0]),
                ],
            ]
            self.door_positions = [
                [-2.65, 2.3, 0],
                [-4.8, -1.75, 0],
                [-6.55, 0.6, 0],
            ]
            self.door_rotations = [
                0.0,
                0.0,
                np.pi / 2.0,
            ]
            self.table_wall_poses = [
                [[-3.18, 0.1, 0.42], 0.0],
                [[-3.18, 0.9, 0.42], 0.0],
            ]
            wall_scales = [
                0.22,
                0.22,
            ]
            self.cabinet_drawer_poses = [
                [[-5.9, 2, 0.5], -np.pi / 2.0],
                [[-4.7, 2, 0.5], -np.pi / 2.0],
                [[-3.5, 2, 0.5], -np.pi / 2.0],
                [[-5.9, 2, 1.8], -np.pi / 2.0],
                [[-4.7, 2, 1.8], -np.pi / 2.0],
                [[-3.5, 2, 1.8], -np.pi / 2.0],
            ]
            self.cabinet_drawer_pose_transform = [
                [
                    0.0,
                    np.array([-0.5, 0, 0]),
                    np.array([-0.5, 0, 0]),
                ],
                [
                    np.pi / 2.0,
                    np.array([-4.2, -4.5, 0]),
                    np.array([-4.2, -4.5, 0]),
                ],
                [
                    -np.pi / 2.0,
                    np.array([-4.3, 5.5, 0]),
                    np.array([-4.3, 5.5, 0]),
                ]
            ]
            self.tabletop_object_name = '036_wood_block'
            self.tabletop_object_scale = 1
            self.tabletop_object_height = 0.91
            self.tabletop_object_dist_tol = 0.1
            self.tabletop_object_interaction_reward = 1.0

            self.tabletop_object_orn = [
                -0.17487982428509494,
                0.6857134620774584,
                -0.6311911117325094,
                0.3175088588585192]

            # self.tabletop_object_pos_range = np.array([
            #     [-4.6, -3.9], [-0.1, 0.2]
            # ])
            self.tabletop_object_pos_range = np.array([
                [-4.1, -4.1], [-0.4, -0.1]
            ])
            # self.tabletop_object_target_range = np.array([
            #     [-4.6, -3.9], [-0.1, 0.2]
            # ])
            self.tabletop_object_target_range = np.array([
                [-4.6, -4.3], [-0.4, -0.1]
            ])
            self.tabletop_object_initial_pos = np.array([0, 0, 0])
            self.tabletop_object_target_pos = np.array([0, 0, 0])
            self.tabletop_reaching_penalty = 1.0
            self.tabletop_reaching_target_height = 0.83
            self.tabletop_reaching_target_range = np.array([
                [-4.6, -4.0], [-0.1, 0.2]
            ])
            self.tabletop_reaching_dist_tol = 0.1
        else:
            if self.arena != 'random_nav':
                assert False, 'model_id unknown'

        if self.arena in ['push_door', 'button_door',
                          'random_manip', 'random_manip_atomic']:
            self.door_axis_link_id = 1
            door_urdf = 'realdoor.urdf' if self.arena in \
                ['push_door', 'random_manip', 'random_manip_atomic'] \
                else 'realdoor_closed.urdf'
            for scale, position, rotation in \
                    zip(door_scales, self.door_positions, self.door_rotations):
                door = InteractiveObj(
                    os.path.join(gibson2.assets_path, 'models',
                                 'scene_components', door_urdf),
                    scale=scale)
                self.simulator.import_articulated_object(door, class_id=2)
                # door pose is fixed
                door.set_position_orientation(position, quatToXYZW(
                    euler2quat(0, 0, rotation), 'wxyz'))
                self.doors.append(door)
                # remove door collision with mesh
                for i in range(p.getNumJoints(door.body_id)):
                    p.setCollisionFilterPair(
                        self.mesh_id, door.body_id, -1, i, 0)
                    p.setCollisionFilterPair(
                        1, door.body_id, -1, i, 0)

            for wall_pose in self.wall_poses:
                wall = InteractiveObj(
                    os.path.join(gibson2.assets_path,
                                 'models',
                                 'scene_components',
                                 'walls_quarter_white.urdf'),
                    scale=0.3)
                self.simulator.import_object(wall, class_id=3)
                # wall pose is fixed
                wall.set_position_orientation(wall_pose[0], wall_pose[1])
                self.walls.append(wall)

            if self.arena == 'button_door':
                self.button_axis_link_id = 1
                self.button_threshold = -0.05
                self.button_reward = 5.0

                self.buttons = []
                for scale in button_scales:
                    button = InteractiveObj(
                        os.path.join(gibson2.assets_path,
                                     'models',
                                     'scene_components',
                                     'eswitch',
                                     'eswitch.urdf'),
                        scale=scale)
                    self.simulator.import_articulated_object(
                        button, class_id=255)
                    self.buttons.append(button)

        elif self.arena == 'obstacles':
            for obstacle_pose in self.obstacle_poses:
                obstacle = BoxShape(pos=obstacle_pose[0],
                                    dim=[0.25, 0.25, 0.6],
                                    mass=10,
                                    color=[1, 0.64, 0, 1])
                self.simulator.import_object(obstacle, class_id=4)
                p.changeDynamics(obstacle.body_id, -1, lateralFriction=0.5)
                self.obstacles.append(obstacle)

        elif self.arena == 'semantic_obstacles':
            for pose, mass, color in \
                    zip(self.semantic_obstacle_poses,
                        self.semantic_obstacle_masses,
                        self.semantic_obstacle_colors):
                obstacle = BoxShape(pos=pose[0],
                                    dim=[self.obstacle_dim,
                                         self.obstacle_dim,
                                         0.6],
                                    mass=mass,
                                    color=color)
                self.simulator.import_object(obstacle, class_id=4)
                p.changeDynamics(obstacle.body_id, -1, lateralFriction=0.5)
                self.obstacles.append(obstacle)

        elif self.arena in ['push_drawers', 'push_chairs',
                            'tabletop_manip', 'tabletop_reaching']:
            # Close off the room with doors
            for scale, position, rotation in \
                    zip(door_scales, self.door_positions, self.door_rotations):
                door = InteractiveObj(
                    os.path.join(gibson2.assets_path, 'models',
                                 'scene_components', 'realdoor_closed.urdf'),
                    scale=scale)
                self.simulator.import_articulated_object(door, class_id=2)
                # door pose is fixed
                door.set_position_orientation(position, quatToXYZW(
                    euler2quat(0, 0, rotation), 'wxyz'))
                for i in range(p.getNumJoints(door.body_id)):
                    p.setCollisionFilterPair(
                        self.mesh_id, door.body_id, -1, i, 0)
                self.doors.append(door)

            if self.arena == 'push_drawers':
                cabinet_models = ['cabinet2/cabinet_0007.urdf'] * 2 + \
                    ['cabinet/cabinet_0004.urdf'] * 2
                for drawer_pose, cabinet_model in \
                        zip(self.cabinet_drawer_poses, cabinet_models):
                    obj = InteractiveObj(filename=os.path.join(
                        gibson2.assets_path, 'models', cabinet_model))
                    self.simulator.import_articulated_object(obj, class_id=30)
                    self.cabinet_drawers.append(obj)
                    obj.set_position_orientation(
                        drawer_pose[0],
                        quatToXYZW(euler2quat(0, 0, drawer_pose[1]), 'wxyz'))

            elif self.arena == 'push_chairs':
                for chair_pose in self.chair_poses:
                    obj = InteractiveObj(filename=os.path.join(
                        gibson2.assets_path,
                        'models/scene_components/chair_and_table/'
                        'alignment_centered_Placida_chair_4_21_03001627.urdf'))
                    # 'free_10_chair_chair_z_up.urdf'))
                    self.simulator.import_articulated_object(obj, class_id=60)
                    self.obstacles.append(obj)
                    obj.set_position_orientation(
                        chair_pose[0],
                        quatToXYZW(euler2quat(0, 0, chair_pose[1]), 'wxyz'))

                obj = InteractiveObj(filename=os.path.join(
                    gibson2.assets_path,
                    'models/scene_components/chair_and_table/'
                    'free_9_table_table_z_up.urdf',
                ), scale=1.2)
                self.simulator.import_articulated_object(obj, class_id=30)
                obj.set_position_orientation(
                    self.table_pose[0],
                    quatToXYZW(euler2quat(
                        0, 0, self.table_pose[1]), 'wxyz'))
                self.table = obj
                self.constraint = p.createConstraint(
                    0, -1, obj.body_id, -1, p.JOINT_FIXED,
                    [0, 0, 1],
                    self.table.get_position(),
                    [0, 0, 0],
                    self.table.get_orientation())

                # Add extra walls to let the LiDAR sense the table
                # TODO: use a different table model
                for wall_pose, wall_scale in \
                        zip(self.table_wall_poses, wall_scales):
                    wall = InteractiveObj(
                        os.path.join(gibson2.assets_path,
                                     'models',
                                     'scene_components',
                                     'walls_quarter_white.urdf'),
                        scale=wall_scale)
                    self.simulator.import_object(wall, class_id=3)
                    # wall pose is fixed
                    wall.set_position_orientation(
                        wall_pose[0],
                        quatToXYZW(euler2quat(0, 0, wall_pose[1]), 'wxyz'))
                    self.walls.append(wall)

            elif self.arena in ['tabletop_manip', 'tabletop_reaching']:
                # load fixed table
                obj = InteractiveObj(filename=os.path.join(
                    gibson2.assets_path,
                    'models/scene_components/chair_and_table/'
                    'free_9_table_table_z_up.urdf',
                ), scale=1.2)
                self.simulator.import_object(obj, class_id=30)
                obj.set_position_orientation(
                    self.table_pose[0],
                    quatToXYZW(euler2quat(
                        0, 0, self.table_pose[1]), 'wxyz'))

                self.table = obj
                # disable collision between table and mesh / floor plane
                p.setCollisionFilterPair(
                    self.mesh_id, self.table.body_id, -1, -1, 0)
                p.setCollisionFilterPair(
                    1, self.table.body_id, -1, -1, 0)

                self.constraint = p.createConstraint(
                    0, -1, obj.body_id, -1, p.JOINT_FIXED,
                    [0, 0, 1],
                    self.table.get_position(),
                    [0, 0, 0],
                    self.table.get_orientation())

                obj = VisualMarker(
                    visual_shape=p.GEOM_CYLINDER,
                    rgba_color=[
                        1, 0, 0, 1],
                    radius=0.1,
                    length=0.02,
                    initial_offset=[0, 0, 0.02 / 2.0])
                self.simulator.import_object(obj, class_id=90)
                self.tabletop_target_marker = obj

                self.robot_constraint = None

                if self.arena == 'tabletop_manip':
                    obj = YCBObject(name=self.tabletop_object_name,
                                    scale=self.tabletop_object_scale)
                    self.simulator.import_object(obj, class_id=40)
                    self.tabletop_object = obj

    def prepare_motion_planner(self):
        self.robot_id = self.robots[0].robot_ids[0]
        self.mesh_id = self.scene.mesh_body_id
        self.map_size = self.scene.trav_map_original_size * \
            self.scene.trav_map_default_resolution

        self.grid_resolution = 128
        self.occupancy_range = 5.0  # m
        robot_footprint_radius = 0.279
        self.robot_footprint_radius_in_map = int(
            robot_footprint_radius / self.occupancy_range *
            self.grid_resolution)

        # self.arm_default_joint_positions = (
        #     0.38548146667743244,
        #     1.1522793897208579,
        #     1.2576467971105596,
        #     -0.312703569911879,
        #     1.7404867100093226,
        #     -0.0962895617312548,
        #     -1.4418232619629425,
        #     -1.6780152866247762
        # )
        # self.arm_default_joint_positions = (
        #     0.02,
        #     np.pi / 2.0,
        #     np.pi / 2.0,
        #     0.0,
        #     np.pi / 2.0,
        #     0.0,
        #     np.pi / 2.0,
        #     0.0
        # )

        if self.config['robot'] == 'Fetch':
            self.arm_default_joint_positions = (0.30322468280792236,
                                                -1.414019864768982,
                                                1.5178184935241699,
                                                0.8189625336474915,
                                                2.200358942909668,
                                                2.9631312579803466,
                                                -1.2862852996643066,
                                                0.0008453550418615341)
            self.arm_joint_ids = joints_from_names(self.robot_id,
                                                   [
                                                       'torso_lift_joint',
                                                       'shoulder_pan_joint',
                                                       'shoulder_lift_joint',
                                                       'upperarm_roll_joint',
                                                       'elbow_flex_joint',
                                                       'forearm_roll_joint',
                                                       'wrist_flex_joint',
                                                       'wrist_roll_joint'
                                                   ])
        elif self.config['robot'] == 'Movo':
            self.arm_default_joint_positions = (0.35, -1.50058731470836, -1.3002625076695704, 0.5204845864369407, \
               -2.6923805472917626, -0.02678584326934146, 0.5065742552588746, \
               -1.562883631882778)
            self.arm_joint_ids = joints_from_names(self.robot_id,
                                                   ["linear_joint",
                                                   "right_shoulder_pan_joint",
                                                    "right_shoulder_lift_joint",
                                                    "right_arm_half_joint",
                                                    "right_elbow_joint",
                                                    "right_wrist_spherical_1_joint",
                                                    "right_wrist_spherical_2_joint",
                                                    "right_wrist_3_joint",
                                                       ])

    def update_action_space(self):
        # action[0] = base_or_arm
        # action[1] = base_subgoal_theta / base_img_v
        # action[2] = base_subgoal_dist / base_img_u
        # action[3] = base_orn
        # action[4] = arm_img_v
        # action[5] = arm_img_u
        # action[6] = arm_push_vector_x
        # action[7] = arm_push_vector_y
        if self.action_map:
            self.base_orn_num_bins = 12  # 12
            self.push_vec_num_bins = 12
            self.downsample_ratio = 4
            self.q_value_size = self.image_height // self.downsample_ratio
            # TODO: assume base and arm Q-value map has the same resolution
            assert self.image_height == self.image_width
            assert self.grid_resolution == self.image_width

            if self.arena == 'random_nav':
                action_dim = self.base_orn_num_bins * (self.q_value_size ** 2)
            elif self.arena in ['random_manip', 'random_manip_atomic',
                                'tabletop_manip', 'tabletop_reaching']:
                action_dim = self.push_vec_num_bins * (self.q_value_size ** 2)
            else:
                action_dim = \
                    self.base_orn_num_bins * (self.q_value_size ** 2) + \
                    self.push_vec_num_bins * (self.q_value_size ** 2)

            self.action_space = gym.spaces.Discrete(action_dim)
        else:
            if self.arena == 'random_nav':
                action_dim = 3
            elif self.arena in ['random_manip',
                                'tabletop_manip', 'tabletop_reaching']:
                action_dim = 4
            elif self.arena == 'random_manip_atomic':
                self.atomic_action_num_directions = 12
                action_dim = 2 + self.atomic_action_num_directions * 2
            else:
                action_dim = 8
            self.action_space = gym.spaces.Box(shape=(action_dim,),
                                               low=-1.0,
                                               high=1.0,
                                               dtype=np.float32)

    def update_observation_space(self):
        if 'occupancy_grid' in self.output:
            self.use_occupancy_grid = True
            occ_grid_dim = 1
            del self.observation_space.spaces['scan']

            if self.draw_path_on_map:
                occ_grid_dim += 1
            if self.draw_objs_on_map:
                occ_grid_dim += 1

            self.observation_space.spaces['occupancy_grid'] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.grid_resolution,
                       self.grid_resolution,
                       occ_grid_dim),
                dtype=np.float32)
        else:
            self.use_occupancy_grid = False

        if self.arena in ['push_drawers', 'push_chairs'] or \
                ('occupancy_grid' in self.output and self.draw_path_on_map):
            self.has_sensor = False
            del self.observation_space.spaces['sensor']
        else:
            self.has_sensor = True

        if self.channel_first:
            for key in ['occupancy_grid', 'rgb', 'depth', 'rgbd']:
                if key in self.output:
                    old_shape = self.observation_space.spaces[key].shape
                    self.observation_space.spaces[key].shape = (
                        old_shape[2], old_shape[0], old_shape[1])

    def update_visualization(self):
        self.base_marker = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                        rgba_color=[1, 0, 0, 1],
                                        radius=0.05,
                                        length=2.0,
                                        initial_offset=[0, 0, 2.0 / 2])
        self.base_marker.load()

        self.arm_marker = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                       rgba_color=[1, 1, 0, 1],
                                       radius=0.1,
                                       length=0.1,
                                       initial_offset=[0, 0, 0.1 / 2])
        self.arm_marker.load()

        self.arm_interact_marker = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                                rgba_color=[1, 0, 1, 1],
                                                radius=0.1,
                                                length=0.1,
                                                initial_offset=[0, 0, 0.1 / 2])
        self.arm_interact_marker.load()

    def plan_base_motion_2d(self, x, y, theta):
        if 'occupancy_grid' in self.output:
            grid = self.occupancy_grid
        elif 'scan' in self.output:
            grid = self.get_local_occupancy_grid(self.state)
        else:
            grid = self.get_local_occupancy_grid()

        yaw = self.robots[0].get_rpy()[2]
        half_occupancy_range = self.occupancy_range / 2.0
        robot_position_xy = self.robots[0].get_position()[:2]
        corners = [
            robot_position_xy + rotate_vector_2d(local_corner, -yaw)
            for local_corner in [
                np.array([half_occupancy_range, half_occupancy_range]),
                np.array([half_occupancy_range, -half_occupancy_range]),
                np.array([-half_occupancy_range, half_occupancy_range]),
                np.array([-half_occupancy_range, -half_occupancy_range]),
            ]
        ]
        path = plan_base_motion_2d(
            self.robot_id,
            [x, y, theta],
            (tuple(np.min(corners, axis=0)), tuple(np.max(corners, axis=0))),
            map_2d=grid,
            occupancy_range=self.occupancy_range,
            grid_resolution=self.grid_resolution,
            robot_footprint_radius_in_map=self.robot_footprint_radius_in_map,
            resolutions=self.base_mp_resolutions,
            obstacles=[],
            algorithm=self.base_mp_algo)

        return path

    def get_local_occupancy_grid(self, state=None):
        assert self.config['robot'] in ['Turtlebot', 'Fetch', 'Movo']
        if self.config['robot'] == 'Turtlebot':
            # Hokuyo URG-04LX-UG01
            laser_linear_range = 5.6
            laser_angular_range = 240.0
            min_laser_dist = 0.05
            laser_link_name = 'scan_link'
        elif self.config['robot'] == 'Fetch' or self.config['robot'] == 'Movo':
            # SICK TiM571-2050101 Laser Range Finder
            laser_linear_range = 25.0
            laser_angular_range = 220.0
            min_laser_dist = 0
            laser_link_name = 'laser_link'

        laser_angular_half_range = laser_angular_range / 2.0
        laser_pose = self.robots[0].parts[laser_link_name].get_pose()
        base_pose = self.robots[0].parts['base_link'].get_pose()

        angle = np.arange(
            np.radians(-laser_angular_half_range),
            np.radians(laser_angular_half_range),
            np.radians(laser_angular_range) / self.n_horizontal_rays
        )
        unit_vector_laser = np.array(
            [[np.cos(ang), np.sin(ang), 0.0] for ang in angle])

        if 'scan' in self.output and state is not None:
            scan = state['scan']
        else:
            scan = self.get_scan()

        scan_laser = unit_vector_laser * \
            (scan * (laser_linear_range - min_laser_dist) + min_laser_dist)

        laser_translation = laser_pose[:3]
        laser_rotation = quat2mat(
            [laser_pose[6], laser_pose[3], laser_pose[4], laser_pose[5]])
        scan_world = laser_rotation.dot(scan_laser.T).T + laser_translation

        base_translation = base_pose[:3]
        base_rotation = quat2mat(
            [base_pose[6], base_pose[3], base_pose[4], base_pose[5]])
        scan_local = base_rotation.T.dot((scan_world - base_translation).T).T
        scan_local = scan_local[:, :2]
        scan_local = np.concatenate(
            [np.array([[0, 0]]), scan_local, np.array([[0, 0]])], axis=0)

        # flip y axis
        scan_local[:, 1] *= -1

        occupancy_grid = np.zeros(
            (self.grid_resolution, self.grid_resolution)).astype(np.uint8)
        scan_local_in_map = scan_local / self.occupancy_range * \
            self.grid_resolution + (self.grid_resolution / 2)
        scan_local_in_map = scan_local_in_map.reshape(
            (1, -1, 1, 2)).astype(np.int32)
        cv2.fillPoly(img=occupancy_grid,
                     pts=scan_local_in_map,
                     color=True,
                     lineType=1)
        cv2.circle(img=occupancy_grid,
                   center=(self.grid_resolution // 2,
                           self.grid_resolution // 2),
                   radius=int(self.robot_footprint_radius_in_map),
                   color=1,
                   thickness=-1)
        return occupancy_grid

    def get_additional_states(self):
        # pos_noise = 0.0
        # cur_pos = self.robots[0].get_position()
        # cur_pos[:2] += np.random.normal(0, pos_noise, 2)

        # rot_noise = 0.0 / 180.0 * np.pi
        # cur_rot = self.robots[0].get_rpy()
        # cur_rot = (cur_rot[0], cur_rot[1], cur_rot[2] +
        #            np.random.normal(0, rot_noise))

        # (mobile) manipulation tasks, no path or goal needed
        if self.arena in ['push_drawers', 'push_chairs',
                          'tabletop_manip', 'tabletop_reaching']:
            waypoints_local_xy = np.zeros(self.scene.num_waypoints * 2)
            target_pos_local_xy = np.zeros(2)
            self.new_potential = 0.0

        # (interactive) navigation tasks, path and goal needed
        else:
            target_pos_local = self.global_to_local(self.target_pos)
            # cache results for reward calculation
            shortest_path, self.new_potential = self.get_shortest_path()

            # geodesic_dist = 0.0
            robot_z = self.robots[0].get_position()[2]

            # closest_idx = np.argmin(np.linalg.norm(
            #     cur_pos[:2] - self.shortest_path, axis=1))
            # shortest_path = self.shortest_path[closest_idx:closest_idx +
            #                                    self.scene.num_waypoints]
            # num_remaining_waypoints = self.scene.num_waypoints - \
            #     shortest_path.shape[0]
            # if num_remaining_waypoints > 0:
            #     remaining_waypoints = np.tile(
            #         self.target_pos[:2], (num_remaining_waypoints, 1))
            #     shortest_path = np.concatenate(
            #         (shortest_path, remaining_waypoints), axis=0)

            shortest_path = np.concatenate(
                (shortest_path,
                 robot_z * np.ones((shortest_path.shape[0], 1))),
                axis=1)

            waypoints_local_xy = np.array([
                self.global_to_local(waypoint)[:2]
                for waypoint in shortest_path]).flatten()
            target_pos_local_xy = target_pos_local[:2]

            # if self.use_occupancy_grid:
            #     waypoints_img_vu = np.zeros_like(waypoints_local_xy)
            #     target_pos_img_vu = np.zeros_like(target_pos_local_xy)

            #     for i in range(self.scene.num_waypoints):
            #         waypoints_img_vu[2 * i] = -waypoints_local_xy[2 * i + 1] \
            #             / (self.occupancy_range / 2.0)
            #         waypoints_img_vu[2 * i + 1] = waypoints_local_xy[2 * i] \
            #             / (self.occupancy_range / 2.0)

            #     target_pos_img_vu[0] = -target_pos_local_xy[1] / \
            #         (self.occupancy_range / 2.0)
            #     target_pos_img_vu[1] = target_pos_local_xy[0] / \
            #         (self.occupancy_range / 2.0)

            #     waypoints_local_xy = waypoints_img_vu
            #     target_pos_local_xy = target_pos_img_vu

        ee_pos_local = self.global_to_local(
            self.robots[0].get_end_effector_position())
        joint_pos = get_joint_positions(self.robot_id, self.arm_joint_ids)
        joint_pos_sin = [np.sin(jp) for jp in joint_pos[1:]]
        joint_pos_cos = [np.cos(jp) for jp in joint_pos[1:]]
        joint_vel = get_joint_velocities(self.robot_id, self.arm_joint_ids)

        additional_states = np.concatenate((waypoints_local_xy,
                                            target_pos_local_xy,
                                            ee_pos_local,
                                            joint_pos,
                                            joint_pos_sin,
                                            joint_pos_cos,
                                            joint_vel,
                                            ))
        # additional_states = np.concatenate((
        #     additional_states,
        #     self.global_to_local(self.tabletop_object_state[0]),
        #     self.global_to_local(self.tabletop_object_target_pos),
        # ))
        # additional_states = np.concatenate((waypoints_local_xy,
        #                                     target_pos_local_xy))
        additional_states = additional_states.astype(np.float32)
        # linear_velocity_local,
        # angular_velocity_local))

        assert len(additional_states) == self.additional_states_dim, \
            'additional states dimension mismatch, {}, {}'.format(
                len(additional_states), self.additional_states_dim)

        return additional_states

    def get_state(self, collision_links=[]):
        state = super(MotionPlanningBaseArmEnv,
                      self).get_state(collision_links)
        if 'occupancy_grid' in self.output:
            occ_grid = self.get_local_occupancy_grid(state)
            self.occupancy_grid = occ_grid
            occ_grid = np.expand_dims(occ_grid, axis=2).astype(np.float32)
            state['occupancy_grid'] = occ_grid
            del state['scan']

            if self.draw_path_on_map:
                waypoints_local_xy = state['sensor'][:(
                    self.scene.num_waypoints * 2)]
                waypoints_img_uv = np.zeros_like(waypoints_local_xy)
                for i in range(self.scene.num_waypoints):
                    waypoints_img_uv[2 * i] = waypoints_local_xy[2 * i] \
                        / self.occupancy_range * self.grid_resolution \
                        + (self.grid_resolution / 2)
                    waypoints_img_uv[2 * i + 1] = \
                        -waypoints_local_xy[2 * i + 1] \
                        / self.occupancy_range * self.grid_resolution \
                        + (self.grid_resolution / 2)
                path_map = np.zeros_like(state['occupancy_grid'])
                for i in range(self.scene.num_waypoints):
                    cv2.circle(img=path_map,
                               center=(int(waypoints_img_uv[2 * i]),
                                       int(waypoints_img_uv[2 * i + 1])),
                               radius=int(self.robot_footprint_radius_in_map),
                               color=1,
                               thickness=-1)
                state['occupancy_grid'] = np.concatenate(
                    (state['occupancy_grid'], path_map), axis=2)

            if self.draw_objs_on_map:
                if self.arena == 'push_chairs':
                    objs = self.obstacles
                elif self.arena == 'push_drawers':
                    objs = self.cabinet_drawers

                objs_local_xy = [
                    self.global_to_local(obj.get_position())[:2]
                    for obj in objs]

                objs_img_uv = np.zeros_like(objs_local_xy)
                for i in range(objs_img_uv.shape[0]):
                    objs_img_uv[i][0] = objs_local_xy[i][0] \
                        / self.occupancy_range * self.grid_resolution \
                        + (self.grid_resolution / 2)
                    objs_img_uv[i][1] = -objs_local_xy[i][1] \
                        / self.occupancy_range * self.grid_resolution \
                        + (self.grid_resolution / 2)
                obj_map = np.zeros_like(state['occupancy_grid'])
                for i in range(objs_img_uv.shape[0]):
                    cv2.circle(img=obj_map,
                               center=(int(objs_img_uv[i][0]),
                                       int(objs_img_uv[i][1])),
                               radius=int(self.robot_footprint_radius_in_map),
                               color=1,
                               thickness=-1)
                state['occupancy_grid'] = np.concatenate(
                    (state['occupancy_grid'], obj_map), axis=2)

            if self.rotate_occ_grid:
                occ_grid_stacked = []
                for n_bin in range(self.base_orn_num_bins):
                    n_bin = (n_bin + 0.5) / self.base_orn_num_bins
                    n_bin = (n_bin * 2 - 1)
                    n_bin = int(np.round(n_bin * 180.0))
                    for i in range(state['occupancy_grid'].shape[2]):
                        img = Image.fromarray(state['occupancy_grid'][:, :, i],
                                              mode='F')
                        img = img.rotate(-n_bin)
                        img = np.asarray(img)
                        occ_grid_stacked.append(img)
                state['occupancy_grid'] = np.stack(occ_grid_stacked,
                                                   axis=2)
        if not self.has_sensor:
            del state['sensor']

        if self.channel_first:
            for key in ['occupancy_grid', 'rgb', 'depth', 'rgbd']:
                if key in self.output:
                    state[key] = state[key].transpose(2, 0, 1)

        self.state = state
        # cv2.imshow('depth', state['depth'])
        # cv2.imshow('scan', state['scan'])
        return state

    def get_geodesic_potential(self):
        return self.new_potential

    # def after_reset_agent(self):
    #     # shortest_path, geodesic_dist = self.get_shortest_path(
    #     #     from_initial_pos=True,
    #     #     entire_path=True)
    #     # # self.shortest_path = shortest_path
    #     _, geodesic_dist = self.get_shortest_path(from_initial_pos=True)
    #     self.new_potential = geodesic_dist

    def get_base_subgoal(self, action):
        """
        Convert action to base_subgoal
        :param action: policy output
        :return: base_subgoal_pos [x, y] in the world frame
        :return: base_subgoal_orn yaw in the world frame
        """
        # if self.use_occupancy_grid:
        #     base_img_v, base_img_u = action[1], action[2]
        #     base_local_y = (-base_img_v) * self.occupancy_range / 2.0
        #     base_local_x = base_img_u * self.occupancy_range / 2.0
        #     base_subgoal_theta = np.arctan2(base_local_y, base_local_x)
        #     base_subgoal_dist = np.linalg.norm([base_local_x, base_local_y])
        # else:
        if self.action_map:
            base_subgoal_theta, base_subgoal_dist = action[1], action[2]
        else:
            base_subgoal_theta = (
                action[1] * 110.0) / 180.0 * np.pi  # [-110.0, 110.0]
            base_subgoal_dist = (action[2] + 1)  # [0.0, 2.0]

        yaw = self.robots[0].get_rpy()[2]
        robot_pos = self.robots[0].get_position()
        base_subgoal_theta += yaw
        base_subgoal_pos = np.array(
            [np.cos(base_subgoal_theta), np.sin(base_subgoal_theta)])
        base_subgoal_pos *= base_subgoal_dist
        base_subgoal_pos = np.append(base_subgoal_pos, 0.0)
        base_subgoal_pos += robot_pos
        base_subgoal_orn = action[3] * np.pi
        base_subgoal_orn += yaw

        self.base_marker.set_position(base_subgoal_pos)

        return base_subgoal_pos, base_subgoal_orn

    def reach_base_subgoal(self, base_subgoal_pos, base_subgoal_orn):
        """
        Attempt to reach base_subgoal and return success / failure
        If failed, reset the base to its original pose
        :param base_subgoal_pos: [x, y] in the world frame
        :param base_subgoal_orn: yaw in the world frame
        :return: whether base_subgoal is achieved
        """
        original_pos = get_base_values(self.robot_id)
        plan_base_start = time.time()
        path = self.plan_base_motion_2d(
            base_subgoal_pos[0], base_subgoal_pos[1], base_subgoal_orn)
        self.episode_metrics['base_mp_time'] += time.time() - plan_base_start
        if path is not None:
            # print('base mp succeeds')
            if self.mode == 'gui':
                for way_point in path:
                    set_base_values_with_z(
                        self.robot_id,
                        [way_point[0],
                         way_point[1],
                         way_point[2]],
                        z=self.initial_height)
                    time.sleep(0.02)
            else:
                set_base_values_with_z(
                    self.robot_id,
                    [base_subgoal_pos[0],
                     base_subgoal_pos[1],
                     base_subgoal_orn],
                    z=self.initial_height)

            # accumupate path length
            self.path_length += \
                np.sum([l2_distance(np.array(path[i][:2]),
                                    np.array(path[i + 1][:2]))
                        for i in range(len(path) - 1)])
            self.episode_metrics['base_mp_success'] += 1
            self.episode_metrics['base_mp_num_waypoints'] += len(path)
            self.episode_metrics['base_mp_path_length'] += \
                np.sum([l2_distance(np.array(path[i]), np.array(path[i + 1]))
                        for i in range(len(path) - 1)])
            return True
        else:
            # print('base mp fails')
            set_base_values_with_z(
                self.robot_id, original_pos, z=self.initial_height)
            self.episode_metrics['base_mp_failure'] += 1
            return False

    def move_base(self, action):
        """
        Execute action for base_subgoal
        :param action: policy output
        :return: whether base_subgoal is achieved
        """
        # print('base')
        # start = time.time()
        base_subgoal_pos, base_subgoal_orn = self.get_base_subgoal(action)
        # self.times['get_base_subgoal'].append(time.time() - start)
        # print('get_base_subgoal', time.time() - start)

        # start = time.time()
        subgoal_success = self.reach_base_subgoal(
            base_subgoal_pos, base_subgoal_orn)
        # self.times['reach_base_subgoal'].append(time.time() - start)
        # print('reach_base_subgoal', time.time() - start)

        return subgoal_success

    def get_arm_subgoal(self, action):
        """
        Convert action to arm_subgoal
        :param action: policy output
        :return: arm_subgoal [x, y, z] in the world frame
        """
        # print('get_arm_subgoal', state['current_step'])
        points = self.simulator.renderer.render_robot_cameras(modes=('3d'))[0]
        height, width = points.shape[0:2]

        arm_img_v = np.clip(int((action[4] + 1) / 2.0 * height), 0, height - 1)
        arm_img_u = np.clip(int((action[5] + 1) / 2.0 * width), 0, width - 1)

        point = points[arm_img_v, arm_img_u]
        camera_pose = (self.robots[0].parts['eyes'].get_pose())
        transform_mat = quat_pos_to_mat(pos=camera_pose[:3],
                                        quat=[camera_pose[6],
                                              camera_pose[3],
                                              camera_pose[4],
                                              camera_pose[5]])
        arm_subgoal = transform_mat.dot(
            np.array([-point[2], -point[0], point[1], 1]))[:3]
        self.arm_marker.set_position(arm_subgoal)
        self.arm_subgoal = arm_subgoal

        push_vector_local = np.array(
            [action[6], action[7]]) * self.arm_interaction_length
        push_vector = rotate_vector_2d(
            push_vector_local, -self.robots[0].get_rpy()[2])
        push_vector = np.append(push_vector, 0.0)
        self.arm_interact_marker.set_position(arm_subgoal + push_vector)

        return arm_subgoal

    def is_collision_free(self, body_a, link_a_list,
                          body_b=None, link_b_list=None):
        """
        :param body_a: body id of body A
        :param link_a_list: link ids of body A that that of interest
        :param body_b: body id of body B (optional)
        :param link_b_list: link ids of body B that are of interest (optional)
        :return: whether the bodies and links of interest are collision-free
        """
        if body_b is None:
            for link_a in link_a_list:
                contact_pts = p.getContactPoints(
                    bodyA=body_a, linkIndexA=link_a)
                if len(contact_pts) > 0:
                    return False
        elif link_b_list is None:
            for link_a in link_a_list:
                contact_pts = p.getContactPoints(
                    bodyA=body_a, bodyB=body_b, linkIndexA=link_a)
                if len(contact_pts) > 0:
                    return False
        else:
            for link_a in link_a_list:
                for link_b in link_b_list:
                    contact_pts = p.getContactPoints(
                        bodyA=body_a, bodyB=body_b,
                        linkIndexA=link_a, linkIndexB=link_b)
                    if len(contact_pts) > 0:
                        return False
        return True

    def get_arm_joint_positions(self, arm_subgoal):
        """
        Attempt to find arm_joint_positions that satisfies arm_subgoal
        If failed, return None
        :param arm_subgoal: [x, y, z] in the world frame
        :return: arm joint positions
        """
        ik_start = time.time()

        max_limits, min_limits, rest_position, joint_range, joint_damping = \
            self.get_ik_parameters()

        n_attempt = 0
        max_attempt = 50
        sample_fn = get_sample_fn(self.robot_id, self.arm_joint_ids)
        base_pose = get_base_values(self.robot_id)

        # find collision-free IK solution for arm_subgoal
        while n_attempt < max_attempt:
            set_joint_positions(self.robot_id, self.arm_joint_ids, sample_fn())
            arm_joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.robots[0].end_effector_part_index(),
                targetPosition=arm_subgoal,
                # targetOrientation=self.robots[0].get_orientation(),
                lowerLimits=min_limits,
                upperLimits=max_limits,
                jointRanges=joint_range,
                restPoses=rest_position,
                jointDamping=joint_damping,
                solver=p.IK_DLS,
                maxNumIterations=100)

            if self.config['robot'] == 'Fetch':
                arm_joint_positions = arm_joint_positions[2:10]
            elif self.config['robot'] == 'Movo':
                arm_joint_positions = arm_joint_positions[:8]

            set_joint_positions(
                self.robot_id, self.arm_joint_ids, arm_joint_positions)

            dist = l2_distance(
                self.robots[0].get_end_effector_position(), arm_subgoal)
            print('dist', dist)
            if dist > self.arm_subgoal_threshold:
                n_attempt += 1
                continue

            # need to simulator_step to get the latest collision
            self.simulator_step()

            # simulator_step will slightly move the robot base and the objects
            set_base_values_with_z(
                self.robot_id, base_pose, z=self.initial_height)
            self.reset_object_states()

            # arm should not have any collision
            collision_free = self.is_collision_free(
                body_a=self.robot_id,
                link_a_list=self.arm_joint_ids)
            if not collision_free:
                n_attempt += 1
                continue

            # gripper should not have any self-collision
            collision_free = self.is_collision_free(
                body_a=self.robot_id,
                link_a_list=[
                    self.robots[0].end_effector_part_index()],
                body_b=self.robot_id)
            if not collision_free:
                n_attempt += 1
                continue

            self.episode_metrics['arm_ik_time'] += time.time() - ik_start
            return arm_joint_positions

        self.episode_metrics['arm_ik_time'] += time.time() - ik_start
        return None

    def reset_obstacles_z(self):
        """
        Make all obstacles perpendicular to the ground
        """
        if self.arena == 'semantic_obstacles':
            obstacle_poses = self.semantic_obstacle_poses
        elif self.arena == 'obstacles':
            obstacle_poses = self.obstacle_poses
        elif self.arena == 'push_chairs':
            obstacle_poses = self.chair_poses
        else:
            assert False

        for obstacle, obstacle_original_pose in \
                zip(self.obstacles, obstacle_poses):
            obstacle_pose = get_base_values(obstacle.body_id)
            set_base_values_with_z(
                obstacle.body_id, obstacle_pose, obstacle_original_pose[0][2])

    def reach_arm_subgoal(self, arm_joint_positions):
        """
        Attempt to reach arm arm_joint_positions and return success / failure
        If failed, reset the arm to its original pose
        :param arm_joint_positions
        :return: whether arm_joint_positions is achieved
        """
        if self.config['robot'] == 'Movo':
            self.robots[0].tuck()

        set_joint_positions(self.robot_id, self.arm_joint_ids,
                            self.arm_default_joint_positions)

        if arm_joint_positions is None:
            self.episode_metrics['arm_ik_failure'] += 1
            return False

        if self.config['robot'] == 'Fetch':
            disabled_collisions = {
                (link_from_name(self.robot_id, 'torso_lift_link'),
                 link_from_name(self.robot_id, 'torso_fixed_link')),
                (link_from_name(self.robot_id, 'torso_lift_link'),
                 link_from_name(self.robot_id, 'shoulder_lift_link')),
                (link_from_name(self.robot_id, 'torso_lift_link'),
                 link_from_name(self.robot_id, 'upperarm_roll_link')),
                (link_from_name(self.robot_id, 'torso_lift_link'),
                 link_from_name(self.robot_id, 'forearm_roll_link')),
                (link_from_name(self.robot_id, 'torso_lift_link'),
                 link_from_name(self.robot_id, 'elbow_flex_link'))}
        elif self.config['robot'] == 'Movo':
            disabled_collisions = {}

        if self.fine_motion_plan:
            self_collisions = True
            mp_obstacles = self.mp_obstacles_id
        else:
            self_collisions = False
            mp_obstacles = []

        plan_arm_start = time.time()
        arm_path = plan_joint_motion(
            self.robot_id,
            self.arm_joint_ids,
            arm_joint_positions,
            disabled_collisions=disabled_collisions,
            self_collisions=self_collisions,
            obstacles=mp_obstacles,
            algorithm=self.arm_mp_algo)
        self.episode_metrics['arm_mp_time'] += time.time() - plan_arm_start
        if arm_path is not None:
            if self.mode == 'gui':
                for joint_way_point in arm_path:
                    set_joint_positions(
                        self.robot_id, self.arm_joint_ids, joint_way_point)
                    time.sleep(0.02)  # animation
            else:
                set_joint_positions(
                    self.robot_id, self.arm_joint_ids, arm_joint_positions)
            self.episode_metrics['arm_mp_success'] += 1
            self.episode_metrics['arm_mp_num_waypoints'] += len(arm_path)
            self.episode_metrics['arm_mp_path_length'] += \
                np.sum([l2_distance(np.array(arm_path[i]),
                                    np.array(arm_path[i + 1]))
                        for i in range(len(arm_path) - 1)])

            return True
        else:
            # print('arm mp fails')
            if self.config['robot'] == 'Movo':
                self.robots[0].tuck()
            set_joint_positions(self.robot_id, self.arm_joint_ids,
                                self.arm_default_joint_positions)
            self.episode_metrics['arm_mp_failure'] += 1
            return False

    def stash_object_states(self):
        if self.arena in ['push_door', 'button_door',
                          'random_manip', 'random_manip_atomic']:
            for i, door in enumerate(self.doors):
                self.door_states[i] = p.getJointState(
                    door.body_id, self.door_axis_link_id)[0]
            if self.arena == 'button_door':
                for i, button in enumerate(self.buttons):
                    self.button_states[i] = p.getJointState(
                        button.body_id, self.button_axis_link_id)[0]
        elif self.arena in ['obstacles', 'semantic_obstacles', 'push_chairs']:
            for i, obstacle in enumerate(self.obstacles):
                self.obstacle_states[i] = obstacle.get_position_orientation()
        elif self.arena == 'push_drawers':
            for body_id in self.cabinet_drawers_states:
                for joint_id in self.cabinet_drawers_states[body_id]:
                    joint_pos = p.getJointState(body_id, joint_id)[0]
                    self.cabinet_drawers_states[body_id][joint_id] = joint_pos
        elif self.arena == 'tabletop_manip':
            self.tabletop_object_state = \
                self.tabletop_object.get_position_orientation()

    def reset_object_states(self):
        """
        Remove any accumulated velocities or forces of objects
        resulting from arm motion planner
        """
        if self.arena in ['push_door', 'button_door',
                          'random_manip', 'random_manip_atomic']:
            for door, door_state in \
                    zip(self.doors, self.door_states):
                p.resetJointState(door.body_id, self.door_axis_link_id,
                                  targetValue=door_state, targetVelocity=0.0)
            if self.arena == 'button_door':
                for button, button_state in \
                        zip(self.buttons, self.button_states):
                    p.resetJointState(button.body_id, self.button_axis_link_id,
                                      targetValue=button_state,
                                      targetVelocity=0.0)
        elif self.arena in ['obstacles', 'semantic_obstacles', 'push_chairs']:
            for obstacle, obstacle_state in \
                    zip(self.obstacles, self.obstacle_states):
                obstacle.set_position_orientation(*obstacle_state)
        elif self.arena == 'push_drawers':
            for body_id in self.cabinet_drawers_states:
                for joint_id in self.cabinet_drawers_states[body_id]:
                    joint_pos = self.cabinet_drawers_states[body_id][joint_id]
                    p.resetJointState(body_id, joint_id,
                                      targetValue=joint_pos,
                                      targetVelocity=0.0)
        elif self.arena == 'tabletop_manip':
            self.tabletop_object.set_position_orientation(
                *self.tabletop_object_state)

    def get_ik_parameters(self):
        if self.config['robot'] == 'Fetch':
            max_limits = [0., 0.] + \
                get_max_limits(self.robot_id, self.arm_joint_ids)
            min_limits = [0., 0.] + \
                get_min_limits(self.robot_id, self.arm_joint_ids)
            # increase torso_lift_joint lower limit to 0.02 to avoid self-collision
            min_limits[2] += 0.02
            rest_position = [0., 0.] + \
                list(get_joint_positions(self.robot_id, self.arm_joint_ids))
            joint_range = list(np.array(max_limits) - np.array(min_limits))
            joint_range = [item + 1 for item in joint_range]
            joint_damping = [0.1 for _ in joint_range]
            
        elif self.config['robot'] == 'Movo':
            max_limits = get_max_limits(self.robot_id, self.robots[0].all_joints)
            min_limits = get_min_limits(self.robot_id, self.robots[0].all_joints)
            rest_position = list(get_joint_positions(self.robot_id, self.robots[0].all_joints))
            joint_range = list(np.array(max_limits) - np.array(min_limits))
            joint_range = [item + 1 for item in joint_range]
            joint_damping = [0.1 for _ in joint_range]
        
        return (
            max_limits, min_limits, rest_position,
            joint_range, joint_damping
        )

    def interact(self, action, arm_subgoal):
        """
        Move the arm according to push_vector and physically
        simulate the interaction
        :param action: policy output
        :param arm_subgoal: starting location of the interaction
        :return: None
        """
        push_vector_local = np.array(
            [action[6], action[7]]) * self.arm_interaction_length
        push_vector = rotate_vector_2d(
            push_vector_local, -self.robots[0].get_rpy()[2])
        push_vector = np.append(push_vector, 0.0)

        max_limits, min_limits, rest_position, joint_range, joint_damping = \
            self.get_ik_parameters()
        base_pose = get_base_values(self.robot_id)

        # # test arm_subgoal + push_vector reachability
        # joint_positions_original = get_joint_positions(
        #     self.robot_id, self.arm_joint_ids)

        # joint_positions = p.calculateInverseKinematics(
        #     self.robot_id,
        #     self.robots[0].parts['gripper_link'].body_part_index,
        #     arm_subgoal + push_vector,
        #     self.robots[0].get_orientation(
        #     ),
        #     lowerLimits=min_limits,
        #     upperLimits=max_limits,
        #     jointRanges=joint_range,
        #     restPoses=rest_position,
        #     jointDamping=joint_damping,
        #     solver=p.IK_DLS,
        #     maxNumIterations=100)[2:10]

        # set_joint_positions(self.robot_id, self.arm_joint_ids,
        #                     joint_positions)
        # diff = l2_distance(arm_subgoal + push_vector,
        #                    self.robots[0].get_end_effector_position())
        # set_joint_positions(self.robot_id, self.arm_joint_ids,
        #                     joint_positions_original)
        # if diff > 0.03:
        #     # print('arm end pose unreachable')
        #     return

        # self.simulator.set_timestep(0.002)
        steps = 50
        for i in range(steps):
            push_goal = np.array(arm_subgoal) + \
                push_vector * (i + 1) / float(steps)

            joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.robots[0].end_effector_part_index(),
                targetPosition=push_goal,
                # targetOrientation=self.robots[0].get_orientation(),
                lowerLimits=min_limits,
                upperLimits=max_limits,
                jointRanges=joint_range,
                restPoses=rest_position,
                jointDamping=joint_damping,
                solver=p.IK_DLS,
                maxNumIterations=100)

            if self.config['robot'] == 'Fetch':
                joint_positions = joint_positions[2:10]
            elif self.config['robot'] == 'Movo':
                joint_positions = joint_positions[:8]

            control_joints(self.robot_id, self.arm_joint_ids, joint_positions)
            if self.config['robot'] == 'Movo':
                self.control_tuck_left()
            self.simulator_step()
            set_base_values_with_z(
                self.robot_id, base_pose, z=self.initial_height)

            if self.arena in \
                    ['obstacles', 'semantic_obstacles', 'push_chairs']:
                self.reset_obstacles_z()

            if self.mode == 'gui':
                time.sleep(0.02)  # for visualization

    def move_arm(self, action):
        """
        Execute action for arm_subgoal and push_vector
        :param action: policy output
        :return: whether arm_subgoal is achieved
        """
        # print('arm')
        # start = time.time()
        arm_subgoal = self.get_arm_subgoal(action)
        # self.times['get_arm_subgoal'].append(time.time() - start)
        # print('get_arm_subgoal', time.time() - start)

        # start = time.time()
        # print(p.getNumBodies())
        # state_id = p.saveState()
        # print('saveState', time.time() - start)

        # start = time.time()
        self.stash_object_states()
        # self.times['stash_object_states'].append(time.time() - start)

        # start = time.time()
        arm_joint_positions = self.get_arm_joint_positions(arm_subgoal)
        # self.times['get_arm_joint_positions'].append(time.time() - start)
        # print('get_arm_joint_positions', time.time() - start)

        # start = time.time()
        subgoal_success = self.reach_arm_subgoal(arm_joint_positions)
        # self.times['reach_arm_subgoal'].append(time.time() - start)
        # print('reach_arm_subgoal', time.time() - start)

        # start = time.time()
        # p.restoreState(stateId=state_id)
        # print('restoreState', time.time() - start)

        # start = time.time()
        self.reset_object_states()
        # self.times['reset_object_states'].append(time.time() - start)

        # print('reset_object_velocities', time.time() - start)

        if subgoal_success:
            # start = time.time()
            self.interact(action, arm_subgoal)
            # self.times['interact'].append(time.time() - start)
            # print('interact', time.time() - start)

        return subgoal_success

    def step(self, action):
        # start = time.time()
        # action[0] = base_or_arm
        # action[1] = base_subgoal_theta
        # action[2] = base_subgoal_dist
        # action[3] = base_orn
        # action[4] = arm_img_v
        # action[5] = arm_img_u
        # action[6] = arm_push_vector_x
        # action[7] = arm_push_vector_y
        # print('-' * 20)
        self.current_step += 1
        if self.action_map:
            # print('action', action)
            assert 0 <= action < self.action_space.n
            new_action = np.zeros(8)
            base_range = self.base_orn_num_bins * (self.q_value_size ** 2)

            if self.arena in ['random_manip', 'random_manip_atomic',
                              'tabletop_manip', 'tabletop_reaching']:
                action += base_range

            # base
            if action < base_range:
                base_orn_bin = action // (self.q_value_size ** 2)
                base_orn_bin = (base_orn_bin + 0.5) / self.base_orn_num_bins
                base_orn_bin = (base_orn_bin * 2 - 1)
                assert -1 <= base_orn_bin <= 1, action

                base_pixel = action % (self.q_value_size ** 2)
                base_pixel_row = base_pixel // self.q_value_size + 0.5
                base_pixel_col = base_pixel % self.q_value_size + 0.5
                if self.rotate_occ_grid:
                    # rotate the pixel back to 0 degree rotation
                    base_pixel_row -= self.q_value_size / 2.0
                    base_pixel_col -= self.q_value_size / 2.0
                    base_pixel_row, base_pixel_col = rotate_vector_2d(
                        np.array([base_pixel_row, base_pixel_col]),
                        -base_orn_bin * np.pi)
                    base_pixel_row += self.q_value_size / 2.0
                    base_pixel_col += self.q_value_size / 2.0

                base_local_y = (-base_pixel_row + self.q_value_size / 2.0) \
                    * (self.occupancy_range / self.q_value_size)
                base_local_x = (base_pixel_col - self.q_value_size / 2.0) \
                    * (self.occupancy_range / self.q_value_size)
                base_subgoal_theta = np.arctan2(base_local_y, base_local_x)
                base_subgoal_dist = np.linalg.norm(
                    [base_local_x, base_local_y])

                new_action[0] = 1.0
                new_action[1] = base_subgoal_theta
                new_action[2] = base_subgoal_dist
                new_action[3] = base_orn_bin

            # arm
            else:
                action -= base_range
                arm_pixel = action % (self.q_value_size ** 2)
                arm_pixel_row = arm_pixel // self.q_value_size + 0.5
                arm_pixel_col = arm_pixel % self.q_value_size + 0.5
                arm_pixel_row = (arm_pixel_row) / self.q_value_size
                arm_pixel_row = arm_pixel_row * 2 - 1
                arm_pixel_col = (arm_pixel_col) / self.q_value_size
                arm_pixel_col = arm_pixel_col * 2 - 1
                assert -1 <= arm_pixel_row <= 1, action
                assert -1 <= arm_pixel_col <= 1, action

                push_vec_bin = action // (self.q_value_size ** 2)
                push_vec_bin = (push_vec_bin + 0.5) / self.push_vec_num_bins
                assert 0 <= push_vec_bin <= 1, action
                push_vec_bin = push_vec_bin * np.pi * 2.0
                push_vector = rotate_vector_2d(
                    np.array([1.0, 0.0]), push_vec_bin)

                new_action[0] = -1.0
                new_action[4] = arm_pixel_row
                new_action[5] = arm_pixel_col
                new_action[6:8] = push_vector

            # print('new_action', new_action)
            action = new_action

        else:
            if self.arena in ['random_nav',
                              'random_manip',
                              'random_manip_atomic',
                              'tabletop_manip',
                              'tabletop_reaching']:
                new_action = np.zeros(8)
                if self.arena == 'random_nav':
                    new_action[0] = 1.0
                    new_action[1:4] = action
                elif self.arena in ['random_manip',
                                    'tabletop_manip', 'tabletop_reaching']:
                    new_action[0] = -1.0
                    new_action[4:8] = action
                elif self.arena == 'random_manip_atomic':
                    new_action[0] = -1.0
                    new_action[4:6] = action[0:2]
                    num_direction_probs = action[2:(
                        2 + self.atomic_action_num_directions)]
                    num_direction_offsets = action[(
                        2 + self.atomic_action_num_directions):]
                    bin_size = np.pi * 2.0 / self.atomic_action_num_directions
                    offset_size = bin_size / 2.0
                    bin_idx = np.argmax(num_direction_probs)
                    bin_center = bin_idx * bin_size
                    push_angle = bin_center + \
                        num_direction_offsets[bin_idx] * offset_size
                    push_vector = rotate_vector_2d(
                        np.array([1.0, 0.0]), -push_angle)
                    new_action[6:8] = push_vector

                action = new_action

        use_base = action[0] > 0.0
        # add action noise before execution
        # action_noise = np.random.normal(0.0, 0.05, action.shape[0] - 1)
        # action[1:] = np.clip(action[1:] + action_noise, -1.0, 1.0)

        self.base_subgoal_success = False
        self.arm_subgoal_success = False
        if use_base:
            subgoal_success = self.move_base(action)
            self.base_subgoal_success = subgoal_success
            # print('move_base:', subgoal_success)
        else:
            subgoal_success = self.move_arm(action)
            self.arm_subgoal_success = subgoal_success
            # print('move_arm:', subgoal_success)

        # start = time.time()
        state, reward, done, info = self.compute_next_step(
            action, use_base, subgoal_success)

        # self.times['compute_next_step'].append(time.time() - start)

        self.step_visualization()
        # print('step time:', time.time() - start)
        return state, reward, done, info

    def get_reward(self, collision_links=[], action=None, info={},
                   base_or_arm='both'):

        reward = 0.0
        if base_or_arm in ['base', 'both']:
            base_reward, info = super(NavigateRandomEnv, self).get_reward(
                collision_links=collision_links, action=action, info=info)

            # ignore navigation reward (assuming no slack reward)
            if self.arena in ['push_drawers', 'push_chairs']:
                base_reward = 0.0

            reward += base_reward

        if base_or_arm in ['arm', 'both']:
            arm_reward = 0.0
            if self.arena == 'button_door':
                button_state = p.getJointState(
                    self.buttons[self.door_idx].body_id,
                    self.button_axis_link_id)[0]
                if not self.door_opened and \
                        button_state < self.button_threshold:
                    print("OPEN DOOR")
                    self.door_opened = True
                    self.doors[self.door_idx].set_position([100.0, 100.0, 0.0])
                    # encourage buttons to be pressed
                    arm_reward = self.button_reward

            elif self.arena in ['push_door',
                                'random_manip', 'random_manip_atomic']:
                new_door_state = p.getJointState(
                    self.doors[self.door_idx].body_id,
                    self.door_axis_link_id)[0]

                # encourage door states to become larger (opening up)
                # TODO: increase push door reward
                door_state_diff = new_door_state - \
                    self.door_states[self.door_idx]
                arm_reward = door_state_diff
                self.door_states[self.door_idx] = new_door_state

                if not self.door_opened and \
                        new_door_state > (60.0 / 180.0 * np.pi):
                    print("PUSH OPEN DOOR")
                    self.door_opened = True
                    if self.arena in ['random_manip', 'random_manip_atomic']:
                        arm_reward += self.success_reward

            elif self.arena in ['obstacles', 'semantic_obstacles']:
                if self.arena == 'semantic_obstacles':
                    obstacle_poses = self.semantic_obstacle_poses
                else:
                    obstacle_poses = self.obstacle_poses

                old_obstacles_moved_dist = 0.0
                new_obstacles_moved_dist = 0.0
                new_obstacle_states = []
                for obstacle, obstacle_state, original_obstacle_pose in \
                        zip(self.obstacles,
                            self.obstacle_states,
                            obstacle_poses):
                    new_obstacle_state = obstacle.get_position_orientation()
                    old_obstacle_pos = np.array(obstacle_state[0][:2])
                    new_obstacle_pos = np.array(new_obstacle_state[0][:2])
                    original_obstacle_pos = np.array(
                        original_obstacle_pose[0][:2])
                    old_obstacles_moved_dist += l2_distance(
                        old_obstacle_pos,
                        original_obstacle_pos)
                    new_obstacles_moved_dist += l2_distance(
                        new_obstacle_pos,
                        original_obstacle_pos)
                    new_obstacle_states.append(new_obstacle_state)

                # encourage obstacles to move away from their original position
                obstacles_moved_dist_diff = (new_obstacles_moved_dist -
                                             old_obstacles_moved_dist)
                arm_reward = (obstacles_moved_dist_diff * 5.0)
                # print('obstacles_moved_dist_diff', obstacles_moved_dist_diff)
                self.obstacle_states = new_obstacle_states

            elif self.arena == 'push_drawers':
                old_drawers_state = 0.0
                new_drawers_state = 0.0
                for body_id in self.cabinet_drawers_states:
                    for joint_id in self.cabinet_drawers_states[body_id]:
                        old_joint_pos = \
                            self.cabinet_drawers_states[body_id][joint_id]
                        new_joint_pos = p.getJointState(body_id, joint_id)[0]
                        old_drawers_state += old_joint_pos
                        new_drawers_state += new_joint_pos
                        self.cabinet_drawers_states[body_id][joint_id] = \
                            new_joint_pos

                # encourage drawers to have smaller joint positions
                # (closing off)
                drawers_diff = old_drawers_state - new_drawers_state
                arm_reward = drawers_diff * 10.0
                if arm_reward > 0.1:
                    print('push drawers reward', arm_reward)

            elif self.arena == 'push_chairs':
                table_pos = np.array(self.table_pose[0][:2])
                old_table_dist = 0.0
                new_table_dist = 0.0
                new_obstacle_states = []
                for obstacle, obstacle_state in \
                        zip(self.obstacles, self.obstacle_states):
                    new_obstacle_state = obstacle.get_position_orientation()
                    old_obstacle_pos = np.array(obstacle_state[0][:2])
                    new_obstacle_pos = np.array(new_obstacle_state[0][:2])
                    old_table_dist += l2_distance(old_obstacle_pos, table_pos)
                    new_table_dist += l2_distance(new_obstacle_pos, table_pos)
                    new_obstacle_states.append(new_obstacle_state)
                self.obstacle_states = new_obstacle_states

                # encourage chairs to come closer to the table
                table_dist_diff = old_table_dist - new_table_dist
                arm_reward = table_dist_diff * 40.0
                if arm_reward > 0.1:
                    print('push chairs reward', arm_reward)
            elif self.arena == 'tabletop_manip':
                old_state = self.tabletop_object_state
                new_state = self.tabletop_object.get_position_orientation()
                old_dist = l2_distance(old_state[0][:2],
                                       self.tabletop_object_target_pos[:2])
                new_dist = l2_distance(new_state[0][:2],
                                       self.tabletop_object_target_pos[:2])
                self.tabletop_object_state = new_state

                # encourage the tabletop object to approach the target pos
                tabletop_dist_diff = old_dist - new_dist
                arm_reward += tabletop_dist_diff * 40.0
                if new_dist < self.tabletop_object_dist_tol:
                    arm_reward += self.success_reward

                # encourage the robot arm to interact with the object
                if l2_distance(old_state[0][:2], new_state[0][:2]) > 1e-3:
                    arm_reward += self.tabletop_object_interaction_reward

            elif self.arena == 'tabletop_reaching':
                if self.continuous_action:
                    arm_position = self.robots[0].get_end_effector_position()
                else:
                    arm_position = self.arm_subgoal
                dist = l2_distance(arm_position,
                                   self.tabletop_reaching_target_pos)
                arm_reward += (-self.tabletop_reaching_penalty) * dist
                if dist < self.tabletop_reaching_dist_tol:
                    arm_reward += self.success_reward

            reward += arm_reward

        return reward, info

    def compute_next_step(self, action, use_base, subgoal_success):
        if not use_base:
            set_joint_positions(self.robot_id, self.arm_joint_ids,
                                self.arm_default_joint_positions)
            if self.config['robot'] == 'Movo':
                self.robots[0].tuck()

        self.simulator.sync()
        state = self.get_state()

        info = {}
        if subgoal_success:
            reward, info = self.get_reward(
                collision_links=[], action=action, info=info,
                base_or_arm='base' if use_base else 'arm')
        else:
            # failed subgoal penalty
            reward = self.failed_subgoal_penalty

        self.episode_return += reward

        done, info = self.get_termination([], action, info)

        if done and self.automatic_reset:
            state = self.reset()

        # self.state['current_step'] = self.current_step
        # print('compute_next_step', self.state['current_step'])
        # print('reward', reward)
        # time.sleep(3)

        return state, reward, done, info

    def reset_initial_and_target_pos(self):
        if self.arena in ['button_door', 'push_door',
                          'obstacles', 'semantic_obstacles',
                          'empty',
                          'random_manip', 'random_manip_atomic',
                          'push_drawers', 'push_chairs',
                          'tabletop_manip', 'tabletop_reaching']:
            floor_height = self.scene.get_floor_height(self.floor_num)

            if self.arena in ['random_manip', 'random_manip_atomic']:
                self.initial_pos = np.array([
                    np.random.uniform(
                        self.initial_pos_range_near_door[0][0],
                        self.initial_pos_range_near_door[0][1]),
                    np.random.uniform(
                        self.initial_pos_range_near_door[1][0],
                        self.initial_pos_range_near_door[1][1]),
                    floor_height
                ])
            else:
                self.initial_pos = np.array([
                    np.random.uniform(
                        self.initial_pos_range[0][0],
                        self.initial_pos_range[0][1]),
                    np.random.uniform(
                        self.initial_pos_range[1][0],
                        self.initial_pos_range[1][1]),
                    floor_height
                ])

            self.initial_height = floor_height + self.initial_pos_z_offset
            # self.robots[0].set_position(pos=[self.initial_pos[0],
            #                                  self.initial_pos[1],
            #                                  self.initial_height])

            if self.arena in ['random_manip', 'random_manip_atomic']:
                self.initial_orn_z = np.random.uniform(
                    self.initial_orn_range_near_door[0],
                    self.initial_orn_range_near_door[1])
            elif self.arena in ['push_drawers', 'push_chairs',
                                'tabletop_manip', 'tabletop_reaching']:
                self.initial_orn_z = np.pi / 2.0
            else:
                self.initial_orn_z = np.random.uniform(-np.pi, np.pi)
                # self.initial_orn_z = np.pi

            self.initial_orn = [0, 0, self.initial_orn_z]

            if self.arena in ['button_door', 'push_door',
                              'random_manip', 'random_manip_atomic']:
                self.door_idx = np.random.randint(0, len(self.doors))
                door_target_pos = self.door_target_pos[self.door_idx]
                self.target_pos = np.array([
                    np.random.uniform(
                        door_target_pos[0][0], door_target_pos[0][1]),
                    np.random.uniform(
                        door_target_pos[1][0], door_target_pos[1][1]),
                    floor_height
                ])
            else:
                self.target_pos = np.array([
                    np.random.uniform(
                        self.target_pos_range[0][0],
                        self.target_pos_range[0][1]),
                    np.random.uniform(
                        self.target_pos_range[1][0],
                        self.target_pos_range[1][1]),
                    floor_height
                ])

        elif self.arena == 'random_nav':
            floor_height = self.scene.get_floor_height(self.floor_num)
            self.initial_height = floor_height + self.initial_pos_z_offset
            super(MotionPlanningBaseArmEnv,
                  self).reset_initial_and_target_pos()

    def before_reset_agent(self):
        if self.arena in ['push_door', 'button_door',
                          'random_manip', 'random_manip_atomic']:
            self.door_opened = False
            self.door_states = np.zeros(len(self.doors))
            for door, angle, pos, orn in \
                    zip(self.doors,
                        self.door_states,
                        self.door_positions,
                        self.door_rotations):
                p.resetJointState(door.body_id, self.door_axis_link_id,
                                  targetValue=angle, targetVelocity=0.0)
                door.set_position_orientation(
                    pos, quatToXYZW(euler2quat(0, 0, orn), 'wxyz'))
            if self.arena == 'button_door':
                self.button_states = np.zeros(len(self.buttons))
                for button, button_pos_range, button_rot, button_state in \
                        zip(self.buttons,
                            self.button_positions,
                            self.button_rotations,
                            self.button_states):
                    button_pos = np.array([
                        np.random.uniform(
                            button_pos_range[0][0], button_pos_range[0][1]),
                        np.random.uniform(
                            button_pos_range[1][0], button_pos_range[1][1]),
                        1.2
                    ])
                    # button pose is randomized
                    button.set_position_orientation(button_pos, quatToXYZW(
                        euler2quat(0, 0, button_rot), 'wxyz'))
                    p.resetJointState(button.body_id, self.button_axis_link_id,
                                      targetValue=button_state,
                                      targetVelocity=0.0)

        elif self.arena in ['obstacles', 'semantic_obstacles']:
            self.obstacle_states = []
            if self.arena == 'obstacles':
                obstacle_poses = self.obstacle_poses
            elif self.arena == 'semantic_obstacles':
                np.random.shuffle(self.semantic_obstacle_poses)
                obstacle_poses = self.semantic_obstacle_poses
            for obstacle, obstacle_pose in zip(self.obstacles, obstacle_poses):
                obstacle.set_position_orientation(*obstacle_pose)
                self.obstacle_states.append(obstacle_pose)

        elif self.arena == 'push_drawers':
            transform = self.cabinet_drawer_pose_transform
            objs = self.cabinet_drawers
            obj_poses = self.cabinet_drawer_poses
            num_obj_poses = len(obj_poses)
            slots = [
                1, 2, 4, 5,
                6, 7, 8, 9, 10, 11,
                12, 15,
            ]
            bottom_slots = [
                1, 2,
                6, 7, 8,
                12
            ]
            pre_selected = [1, 2, 4, 5]
            for i, obj in enumerate(self.cabinet_drawers):
                if self.randomize_object_pose:
                    # the first and second drawer cabinets stay at the bottom
                    if i < 2:
                        selected_slot = np.random.choice(bottom_slots)
                    else:
                        selected_slot = np.random.choice(slots)
                else:
                    selected_slot = pre_selected[i]
                selected_pose = selected_slot % num_obj_poses
                selected_transform = selected_slot // num_obj_poses

                obj_pose = copy.deepcopy(obj_poses[selected_pose])
                # dirty hack to make drawer cabinets attached to the wall
                if i < 2:
                    obj_pose[0][1] -= 0.2
                rot, trans_low, trans_high = transform[selected_transform]
                trans = np.random.uniform(trans_low, trans_high)

                pos = rotate_vector_3d(obj_pose[0], 0, 0, rot) + trans
                orn = quatToXYZW(euler2quat(
                    0, 0, obj_pose[1] - rot), 'wxyz')
                obj.set_position_orientation(pos, orn)
                slots.remove(selected_slot)
                if selected_slot in bottom_slots:
                    bottom_slots.remove(selected_slot)

            self.cabinet_drawers_states = dict()
            for obj in self.cabinet_drawers:
                body_id = obj.body_id
                self.cabinet_drawers_states[body_id] = dict()
                for joint_id in range(p.getNumJoints(body_id)):
                    _, _, joint_type, _, _, _, _, _, \
                        lower, upper, _, _, _, _, _, _, _ = p.getJointInfo(
                            body_id, joint_id)
                    if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                        jp = np.random.uniform(lower, upper)
                        p.resetJointState(body_id, joint_id,
                                          targetValue=jp,
                                          targetVelocity=0)
                        self.cabinet_drawers_states[body_id][joint_id] = jp

        elif self.arena == 'push_chairs':
            transform = self.chair_pose_transform
            objs = self.obstacles
            obj_poses = self.chair_poses
            unique_obj = self.table
            unique_obj_pose = self.table_pose

            self.obstacle_states = []
            rot, trans_low, trans_high = \
                transform[np.random.randint(len(transform))]
            trans = np.random.uniform(trans_low, trans_high)
            for obstacle, obstacle_pose in zip(objs, obj_poses):
                if self.randomize_object_pose:
                    pos = rotate_vector_3d(obstacle_pose[0], 0, 0, rot) + trans
                    orn = quatToXYZW(euler2quat(
                        0, 0, obstacle_pose[1] - rot), 'wxyz')
                else:
                    pos = obstacle_pose[0]
                    orn = quatToXYZW(euler2quat(
                        0, 0, obstacle_pose[1]), 'wxyz')
                obstacle.set_position_orientation(pos, orn)
                self.obstacle_states.append((pos, orn))

            if self.randomize_object_pose:
                pos = rotate_vector_3d(unique_obj_pose[0], 0, 0, rot) + trans
                orn = quatToXYZW(euler2quat(
                    0, 0, unique_obj_pose[1] - rot), 'wxyz')
            else:
                pos = unique_obj_pose[0]
                orn = quatToXYZW(euler2quat(
                    0, 0, unique_obj_pose[1]), 'wxyz')
            unique_obj.set_position_orientation(pos, orn)

            p.removeConstraint(self.constraint)
            self.constraint = p.createConstraint(0, -1, unique_obj.body_id,
                                                 -1, p.JOINT_FIXED,
                                                 [0, 0, 1],
                                                 unique_obj.get_position(),
                                                 [0, 0, 0],
                                                 unique_obj.get_orientation())

            for obstacle, obstacle_pose in \
                    zip(self.walls, self.table_wall_poses):
                if self.randomize_object_pose:
                    pos = rotate_vector_3d(
                        obstacle_pose[0], 0, 0, rot) + trans
                    orn = quatToXYZW(euler2quat(
                        0, 0, obstacle_pose[1] - rot), 'wxyz')
                else:
                    pos = obstacle_pose[0]
                    orn = quatToXYZW(euler2quat(
                        0, 0, obstacle_pose[1]), 'wxyz')
                obstacle.set_position_orientation(pos, orn)

        elif self.arena == 'tabletop_manip':
            if self.randomize_object_pose:
                self.tabletop_object_initial_pos = np.array([
                    np.random.uniform(
                        self.tabletop_object_pos_range[0][0],
                        self.tabletop_object_pos_range[0][1]),
                    np.random.uniform(
                        self.tabletop_object_pos_range[1][0],
                        self.tabletop_object_pos_range[1][1]),
                    self.tabletop_object_height
                ])

                dist = 0
                # initial distance has to be >2x distance tolerance
                while dist < (self.tabletop_object_dist_tol * 2.0):
                    self.tabletop_object_target_pos = np.array([
                        np.random.uniform(
                            self.tabletop_object_target_range[0][0],
                            self.tabletop_object_target_range[0][1]),
                        np.random.uniform(
                            self.tabletop_object_target_range[1][0],
                            self.tabletop_object_target_range[1][1]),
                        self.tabletop_object_height
                    ])
                    dist = l2_distance(self.tabletop_object_target_pos,
                                       self.tabletop_object_initial_pos)
            else:
                self.tabletop_object_initial_pos = np.array([
                    -4.3, -0.14, self.tabletop_object_height
                ])
                self.tabletop_object_target_pos = np.array([
                    -4.5, -0.3, self.tabletop_object_height
                ])

            self.tabletop_object.set_position_orientation(
                self.tabletop_object_initial_pos, self.tabletop_object_orn)
            self.tabletop_target_marker.set_position(
                [self.tabletop_object_target_pos[0],
                 self.tabletop_object_target_pos[1],
                 self.tabletop_object_height - 0.1]
            )
            self.tabletop_object_state = (self.tabletop_object_initial_pos,
                                          self.tabletop_object_orn)
            if self.robot_constraint is not None:
                p.removeConstraint(self.robot_constraint)

        elif self.arena == 'tabletop_reaching':
            self.tabletop_reaching_target_pos = np.array([
                np.random.uniform(
                    self.tabletop_reaching_target_range[0][0],
                    self.tabletop_reaching_target_range[0][1]),
                np.random.uniform(
                    self.tabletop_reaching_target_range[1][0],
                    self.tabletop_reaching_target_range[1][1]),
                self.tabletop_reaching_target_height,
            ])
            self.tabletop_target_marker.set_position(
                self.tabletop_reaching_target_pos)
            if self.robot_constraint is not None:
                p.removeConstraint(self.robot_constraint)

    def after_reset_agent(self):
        if self.arena in ['obstacles', 'semantic_obstacles', 'push_chairs']:
            for obstacle, obstacle_pose in \
                    zip(self.obstacles, self.obstacle_states):
                current_pos = obstacle.get_position()
                original_pos = obstacle_pose[0]
                dist = l2_distance(np.array(current_pos),
                                   np.array(original_pos))
                if dist > 0.05:
                    print('obstacle pose is >0.05m above the ground')
        elif self.arena in ['tabletop_manip', 'tabletop_reaching']:
            self.robot_constraint = \
                p.createConstraint(0, -1, self.robots[0].robot_ids[0],
                                   -1, p.JOINT_FIXED,
                                   [0, 0, 1],
                                   self.robots[0].get_position(),
                                   [0, 0, 0],
                                   self.robots[0].get_orientation())

    # def reset(self):
    #     self.state = super(MotionPlanningBaseArmEnv, self).reset()
    #     self.geodesic_dist = self.get_shortest_path(from_initial_pos=True)[1]
    #     self.state['current_step'] = self.current_step
    #     return self.state

    def get_termination(self, collision_links=[], action=None, info={}):
        done, info = super(MotionPlanningBaseArmEnv,
                           self).get_termination(collision_links, action, info)

        if self.arena in ['random_manip', 'random_manip_atomic']:
            new_door_state = p.getJointState(
                self.doors[self.door_idx].body_id, self.door_axis_link_id)[0]
            if new_door_state > (60.0 / 180.0 * np.pi):
                done = True
                info['success'] = True

        elif self.arena == 'tabletop_manip':
            tabletop_object_pos = self.tabletop_object.get_position()
            dist = l2_distance(
                tabletop_object_pos[:2],
                self.tabletop_object_target_pos[:2])
            if dist < self.tabletop_object_dist_tol:
                done = True
                info['success'] = True

            # the object drops from the table
            if tabletop_object_pos[2] < self.tabletop_object_height - 0.2:
                done = True
                info['success'] = False

        elif self.arena == 'tabletop_reaching':
            if self.continuous_action or self.arm_subgoal_success:
                if self.continuous_action:
                    arm_position = self.robots[0].get_end_effector_position()
                else:
                    arm_position = self.arm_subgoal
                dist = l2_distance(arm_position,
                                   self.tabletop_reaching_target_pos)
                if dist < self.tabletop_reaching_dist_tol:
                    done = True
                    info['success'] = True
        if done:
            self.compute_metrics(info)

        return done, info

    def compute_metrics(self, info):
        if self.arena == 'tabletop_manip':
            dist_to_goal = l2_distance(
                self.tabletop_object.get_position()[:2],
                self.tabletop_object_target_pos[:2])
        elif self.arena == 'tabletop_reaching':
            if self.continuous_action:
                arm_position = self.robots[0].get_end_effector_position()
            else:
                arm_position = self.arm_subgoal
            dist_to_goal = l2_distance(arm_position,
                                       self.tabletop_reaching_target_pos)
        else:
            dist_to_goal = self.get_geodesic_potential()
        self.episode_metrics['dist_to_goal'] = dist_to_goal

        if self.arena in ['button_door', 'push_door']:
            self.episode_metrics['doors_opened'] = self.door_opened
        elif self.arena == 'push_drawers':
            drawers_closed_5 = 0
            drawers_closed_10 = 0
            for body_id in self.cabinet_drawers_states:
                for joint_id in self.cabinet_drawers_states[body_id]:
                    joint_pos = self.cabinet_drawers_states[body_id][joint_id]
                    joint_type = p.getJointInfo(body_id, joint_id)[2]
                    if joint_type == p.JOINT_REVOLUTE:
                        drawers_closed_5 += int(joint_pos < np.deg2rad(5.0))
                        drawers_closed_10 += int(joint_pos < np.deg2rad(10.0))
                    else:
                        drawers_closed_5 += int(joint_pos < 0.05)
                        drawers_closed_10 += int(joint_pos < 0.10)
            self.episode_metrics['drawers_closed_5'] = drawers_closed_5
            self.episode_metrics['drawers_closed_10'] = drawers_closed_10
        elif self.arena == 'push_chairs':
            chairs_pushed_5 = 0
            chairs_pushed_10 = 0
            # minimum distance is 0.4 because table has width
            min_dist = 0.4
            table_pos = np.array(self.table_pose[0][:2])
            for obstacle_state in self.obstacle_states:
                obstacle_pos = np.array(obstacle_state[0][:2])
                chairs_pushed_5 += \
                    int(l2_distance(obstacle_pos, table_pos) < 0.05 + min_dist)
                chairs_pushed_10 += \
                    int(l2_distance(obstacle_pos, table_pos) < 0.1 + min_dist)
            self.episode_metrics['chairs_pushed_5'] = chairs_pushed_5
            self.episode_metrics['chairs_pushed_10'] = chairs_pushed_10

        for key in self.metric_keys:
            if key in ['episode_return', 'episode_length', 'collision_step',
                       'path_length', 'geodesic_dist', 'success', 'spl']:
                self.metrics[key].append(info[key])
            else:
                self.metrics[key].append(self.episode_metrics[key])

        self.episode_metrics = {
            key: 0.0 for key in self.metric_keys
        }

        if self.log_dir is not None:
            self.logger.info('current_episode: ' + str(self.current_episode))
            for key in self.metrics:
                self.logger.info(key + ': ' + str(np.mean(self.metrics[key])))


class MotionPlanningBaseArmContinuousEnv(MotionPlanningBaseArmEnv):
    def __init__(self,
                 config_file,
                 model_id=None,
                 collision_reward_weight=0.0,
                 mode='headless',
                 action_timestep=1 / 10.0,
                 physics_timestep=1 / 240.0,
                 device_idx=0,
                 random_height=False,
                 automatic_reset=False,
                 arena=None,
                 log_dir=None,
                 ):
        if arena in ['tabletop_manip', 'tabletop_reaching']:
            # needs more accurate physics simulation
            physics_timestep = min(physics_timestep, 1 / 100.0)

        super(MotionPlanningBaseArmContinuousEnv, self).__init__(
            config_file,
            model_id=model_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            automatic_reset=automatic_reset,
            random_height=random_height,
            device_idx=device_idx,
            arena=arena,
            collision_reward_weight=collision_reward_weight,
            log_dir=log_dir,
        )

        self.continuous_action = True

        # revert back to raw action space
        if self.arena == 'random_nav':
            self.action_space = gym.spaces.Box(
                shape=(self.robots[0].wheel_dim,),
                low=-1.0,
                high=1.0,
                dtype=np.float32)
        elif self.arena in ['random_manip', 'random_manip_atomic',
                            'tabletop_manip', 'tabletop_reaching']:
            self.action_space = gym.spaces.Box(
                shape=(self.robots[0].arm_dim,),
                low=-1.0,
                high=1.0,
                dtype=np.float32)
        else:
            self.action_space = self.robots[0].action_space

        if self.arena in ['tabletop_manip', 'tabletop_reaching']:
            if self.arena == 'tabletop_manip':
                self.robots[0].arm_default_joint_positions = (
                    0.30322468280792236, -np.pi / 2.0, np.pi / 4.0,
                    np.pi / 2.0, np.pi / 2.0, -np.pi / 4.0, 0, 0)
                self.tabletop_object_interaction_reward /= 30.0
            elif self.arena == 'tabletop_reaching':
                self.robots[0].arm_default_joint_positions = (
                    0.30322468280792236, 0, 0, 0, np.pi / 6.0, 0, 0, 0)
                self.tabletop_reaching_penalty /= 30.0

            # reduce arm maximum velocity by half
            self.robots[0].action_high[-self.robots[0].arm_dim:] *= 0.5
            self.robots[0].action_low[-self.robots[0].arm_dim:] *= 0.5

    def step(self, action):
        if self.arena == 'random_nav':
            new_action = np.zeros(10)
            new_action[:2] = action
            action = new_action
        elif self.arena in ['random_manip', 'random_manip_atomic',
                            'tabletop_manip', 'tabletop_reaching']:
            new_action = np.zeros(10)
            new_action[3:] = action
            action = new_action

        return super(NavigateRandomEnv, self).step(action)

    def get_reward(self, collision_links=[], action=None, info={}):
        reward, info = super(
            MotionPlanningBaseArmContinuousEnv, self).get_reward(
            collision_links=collision_links, action=action, info=info,
            base_or_arm='both')

        # zero out reward if last step has collision
        collision_links_flatten = [
            item for sublist in collision_links for item in sublist]
        if len(collision_links_flatten) > 0:
            reward = 0.0

        return reward, info


class MotionPlanningBaseArmHRL4INEnv(MotionPlanningBaseArmContinuousEnv):
    def __init__(self,
                 config_file,
                 model_id=None,
                 collision_reward_weight=0.0,
                 mode='headless',
                 action_timestep=1 / 10.0,
                 physics_timestep=1 / 240.0,
                 device_idx=0,
                 random_height=False,
                 automatic_reset=False,
                 arena=None,
                 log_dir=None,
                 ):
        super(MotionPlanningBaseArmHRL4INEnv, self).__init__(
            config_file,
            model_id=model_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            automatic_reset=automatic_reset,
            random_height=random_height,
            device_idx=device_idx,
            arena=arena,
            collision_reward_weight=collision_reward_weight,
            log_dir=log_dir,
        )
        self.observation_space.spaces['arm_world'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32)
        self.observation_space.spaces['yaw'] = gym.spaces.Box(
            low=-np.pi * 2,
            high=np.pi * 2,
            shape=(),
            dtype=np.float32)

    def get_state(self, collision_links=[]):
        state = super(MotionPlanningBaseArmHRL4INEnv,
                      self).get_state(collision_links)
        state['arm_world'] = self.robots[0].get_end_effector_position()
        state['yaw'] = self.robots[0].get_rpy()[2]
        return state

    def set_subgoal(self, ideal_next_state):
        self.arm_marker.set_position(ideal_next_state)
        self.base_marker.set_position(
            [ideal_next_state[0], ideal_next_state[1], 0])

    def set_subgoal_type(self, only_base=True):
        if only_base:
            # Make the marker for the end effector completely transparent
            self.arm_marker.set_color([0, 0, 0, 0.0])
        else:
            self.arm_marker.set_color([0, 0, 0, 0.8])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')

    parser.add_argument('--arena',
                        '-a',
                        choices=['button_door', 'push_door',
                                 'obstacles', 'semantic_obstacles',
                                 'empty', 'random_nav',
                                 'random_manip', 'random_manip_atomic',
                                 'push_drawers', 'push_chairs',
                                 'tabletop_manip', 'tabletop_reaching'],
                        default='push_door',
                        help='which arena to use (default: push_door)')

    parser.add_argument('--action_type',
                        '-t',
                        choices=['high-level', 'low-level', 'hrl4in'],
                        default='high-level',
                        help='which action type to use (default: high-level)')

    args = parser.parse_args()

    if args.action_type == 'high-level':
        nav_env = MotionPlanningBaseArmEnv(config_file=args.config,
                                           mode=args.mode,
                                           action_timestep=1 / 500.0,
                                           physics_timestep=1 / 500.0,
                                           arena=args.arena,
                                           action_map=False,
                                           channel_first=True,
                                           draw_path_on_map=False,
                                           draw_objs_on_map=False,
                                           rotate_occ_grid=False,
                                           randomize_object_pose=True,
                                           )
    elif args.action_type == 'low-level':
        nav_env = MotionPlanningBaseArmContinuousEnv(config_file=args.config,
                                                     mode=args.mode,
                                                     action_timestep=1 / 10.0,
                                                     physics_timestep=1 / 40.0,
                                                     arena=args.arena,
                                                     )
    else:
        nav_env = MotionPlanningBaseArmHRL4INEnv(config_file=args.config,
                                                 mode=args.mode,
                                                 action_timestep=1 / 10.0,
                                                 physics_timestep=1 / 40.0,
                                                 arena=args.arena,
                                                 )

    for episode in range(100):
        print('Episode: {}'.format(episode))
        episode_return = 0.0
        state = nav_env.reset()
        embed()
        start = time.time()
        for i in range(10000000):
            print('Step: {}'.format(i))
            action = nav_env.action_space.sample()
            embed()
            state, reward, done, info = nav_env.step(action)
            episode_return += reward
            print('Reward:', reward)
            # embed()
            # time.sleep(0.05)
            # nav_env.step()
            # for step in range(50):  # 500 steps, 50s world time
            #    action = nav_env.action_space.sample()
            #    state, reward, done, _ = nav_env.step(action)
            #    # print('reward', reward)
            if done:
                print('Episode return:', episode_return)
                print('Episode length:', info['episode_length'])
                break
        print("Time", time.time() - start)

    for key in nav_env.times:
        print(key, len(nav_env.times[key]), np.sum(
            nav_env.times[key]), np.mean(nav_env.times[key]))

    nav_env.clean()
