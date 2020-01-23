from gibson2.core.physics.interactive_objects import VisualMarker, InteractiveObj, BoxShape
import gibson2
from gibson2.utils.utils import parse_config, rotate_vector_3d, rotate_vector_2d, l2_distance, quatToXYZW
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
from gibson2.core.render.utils import quat_pos_to_mat
from gibson2.external.pybullet_tools.utils import set_base_values, joint_from_name, set_joint_position, \
    set_joint_positions, add_data_path, connect, plan_base_motion, plan_joint_motion, enable_gravity, \
    joint_controller, dump_body, load_model, joints_from_names, user_input, disconnect, get_joint_positions, \
    get_link_pose, link_from_name, HideOutput, get_pose, wait_for_user, dump_world, plan_nonholonomic_motion, \
    set_point, create_box, stable_z, control_joints, get_max_limits, get_min_limits, get_base_values, \
    plan_base_motion_2d, get_sample_fn, add_p2p_constraint, remove_constraint, set_base_values_with_z


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
                 eval=False
                 ):
        super(MotionPlanningEnv, self).__init__(config_file,
                                                model_id=model_id,
                                                mode=mode,
                                                action_timestep=action_timestep,
                                                physics_timestep=physics_timestep,
                                                automatic_reset=automatic_reset,
                                                random_height=False,
                                                device_idx=device_idx)

        self.mp_loaded = False
        # override some parameters:
        self.max_step = 20
        self.planner_step = 0
        self.action_space = gym.spaces.Box(shape=(3,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

        self.eval = eval

    def prepare_motion_planner(self):
        self.robot_id = self.robots[0].robot_ids[0]
        self.mesh_id = self.scene.mesh_body_id
        self.map_size = self.scene.trav_map_original_size * self.scene.trav_map_default_resolution
        print(self.robot_id, self.mesh_id, self.map_size)
        self.marker = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                   rgba_color=[1, 0, 0, 1],
                                   radius=0.1,
                                   length=0.1,
                                   initial_offset=[0, 0, 0.1 / 2.0])
        self.marker.load()
        self.mp_loaded = True

    def plan_base_motion(self, x, y, theta):
        half_size = self.map_size / 2.0
        # if self.mode == 'gui':
        #     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        path = plan_base_motion(
            self.robot_id,
            [x, y, theta],
            ((-half_size, -half_size), (half_size, half_size)),
            obstacles=[self.mesh_id])
        # if self.mode == 'gui':
        #     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        return path

    def step(self, pt):
        # point = [x,y]
        x = int((pt[0] + 1) / 2.0 * 150)
        y = int((pt[1] + 1) / 2.0 * 128)
        yaw = self.robots[0].get_rpy()[2]
        orn = pt[2] * np.pi + yaw

        opos = get_base_values(self.robot_id)

        self.get_additional_states()
        org_potential = self.get_potential()

        if x < 128:
            state, reward, done, _ = super(MotionPlanningEnv, self).step([0, 0])

            points = state['pc']
            point = points[x, y]

            camera_pose = (self.robots[0].parts['eyes'].get_pose())
            transform_mat = quat_pos_to_mat(pos=camera_pose[:3],
                                            quat=[camera_pose[6], camera_pose[3], camera_pose[4], camera_pose[5]])

            projected_point = (transform_mat).dot(np.array([-point[2], -point[0], point[1], 1]))

            subgoal = projected_point[:2]
        else:
            subgoal = list(opos)[:2]

        path = self.plan_base_motion(subgoal[0], subgoal[1], orn)
        if path is not None:
            self.marker.set_position([subgoal[0], subgoal[1], 0.1])
            if not self.eval:
                bq = path[-1]
                set_base_values(self.robot_id, [bq[0], bq[1], bq[2]])
            else:
                for bq in path:
                    set_base_values(self.robot_id, [bq[0], bq[1], bq[2]])
                    time.sleep(0.02)  # for visualization
            state, _, done, info = super(MotionPlanningEnv, self).step([0, 0])
            self.get_additional_states()
            reward = org_potential - self.get_potential()
        else:
            set_base_values(self.robot_id, opos)
            state, _, done, info = super(MotionPlanningEnv, self).step([0, 0])
            reward = -0.02

        done = False

        if l2_distance(self.target_pos, self.robots[0].get_position()) < self.dist_tol:
            reward += self.success_reward  # |success_reward| = 10.0 per step
            done = True
        else:
            done = False

        print('reward', reward)

        self.planner_step += 1

        if self.planner_step > self.max_step:
            done = True
        # print(info)
        # if info['success']:
        #    done = True
        info['planner_step'] = self.planner_step
        del state['pc']

        return state, reward, done, info

    def reset(self):
        state = super(MotionPlanningEnv, self).reset()
        if not self.mp_loaded:
            self.prepare_motion_planner()

        self.planner_step = 0

        del state['pc']

        return state


class MotionPlanningBaseArmEnv(NavigateRandomEnv):
    def __init__(self,
                 config_file,
                 model_id=None,
                 collision_reward_weight=0.0,
                 mode='headless',
                 action_timestep=1 / 10.0,
                 physics_timestep=1 / 240.0,
                 device_idx=0,
                 automatic_reset=False,
                 eval=False,
                 arena=None,
                 ):
        super(MotionPlanningBaseArmEnv, self).__init__(config_file,
                                                       model_id=model_id,
                                                       mode=mode,
                                                       action_timestep=action_timestep,
                                                       physics_timestep=physics_timestep,
                                                       automatic_reset=automatic_reset,
                                                       random_height=False,
                                                       device_idx=device_idx)

        # # real sensor spec for Fetch
        # resolution = self.config.get('resolution', 64)
        # width = resolution
        # height = int(width * (480.0 / 640.0))
        # if 'rgb' in self.output:
        #     self.observation_space.spaces['rgb'] = gym.spaces.Box(low=0.0,
        #                                                           high=1.0,
        #                                                           shape=(height, width, 3),
        #                                                           dtype=np.float32)
        # if 'depth' in self.output:
        #     self.observation_space.spaces['depth'] = gym.spaces.Box(low=0.0,
        #                                                             high=1.0,
        #                                                             shape=(height, width, 1),
        #                                                             dtype=np.float32)

        self.arena = arena
        self.eval = eval
        self.visualize_waypoints = True
        if self.visualize_waypoints and self.mode == 'gui':
            cyl_length = 0.2
            self.waypoints_vis = [VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                               rgba_color=[0, 1, 0, 0.3],
                                               radius=0.1,
                                               length=cyl_length,
                                               initial_offset=[0, 0, cyl_length / 2.0]) for _ in range(1000)]
            for waypoint in self.waypoints_vis:
                waypoint.load()

        self.new_potential = None
        self.collision_reward_weight = collision_reward_weight

        # action[0] = base_or_arm
        # action[1] = base_subgoal_theta
        # action[2] = base_subgoal_dist
        # action[3] = base_orn
        # action[4] = arm_img_u
        # action[5] = arm_img_v
        # action[6] = arm_push_vector_x
        # action[7] = arm_push_vector_y
        self.action_space = gym.spaces.Box(shape=(8,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.prepare_motion_planner()

        self.base_marker = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                        rgba_color=[1, 0, 0, 1],
                                        radius=0.1,
                                        length=2.0,
                                        initial_offset=[0, 0, 2.0 / 2.0])
        self.base_marker.load()

        self.arm_marker = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                       rgba_color=[1, 1, 0, 1],
                                       radius=0.1,
                                       length=0.1,
                                       initial_offset=[0, 0, 0.1 / 2.0])
        self.arm_marker.load()
        self.arm_default_joint_positions = (0.38548146667743244, 1.1522793897208579,
                                            1.2576467971105596, -0.312703569911879,
                                            1.7404867100093226, -0.0962895617312548,
                                            -1.4418232619629425, -1.6780152866247762)
        self.arm_joint_ids = joints_from_names(self.robot_id,
                                               ['torso_lift_joint',
                                                'shoulder_pan_joint',
                                                'shoulder_lift_joint',
                                                'upperarm_roll_joint',
                                                'elbow_flex_joint',
                                                'forearm_roll_joint',
                                                'wrist_flex_joint',
                                                'wrist_roll_joint'])
        self.arm_subgoal_threshold = 0.05
        self.failed_subgoal_penalty = -0.0

        self.prepare_scene()

    def prepare_scene(self):
        if self.scene.model_id == 'Avonia':
            door_scales = [1.0, 0.9]
            self.door_positions = [[-3.5, 0, 0.0], [-1.2, -2.47, 0.0]]
            self.door_rotations = [np.pi / 2.0, -np.pi / 2.0]
            wall_poses = [
                [[-3.5, 0.45, 0.45], quatToXYZW(euler2quat(0, 0, np.pi / 2.0), 'wxyz')],
                [[-3.5, -0.4, 0.45], quatToXYZW(euler2quat(0, 0, -np.pi / 2.0), 'wxyz')],
            ]
            self.door_target_pos = [
                [[-5.5, -4.5], [-1.0, 1.0]],
                [[0.5, 2.0], [-4.5, -3.0]]
            ]

            button_scales = [2.0, 2.0]
            self.button_positions = [
                [[-2.85, -2.85], [1.2, 1.7]],
                [[-2.1, -1.6], [-2.95, -2.95]]
            ]
            self.button_rotations = [-np.pi / 2.0, 0.0]

            self.obstacle_poses = [[-3.5, 0.5, 0.6], [-3.5, -0.05, 0.6], [-3.5, -0.6, 0.6], [-3.5, -1.15, 0.6]]

            # TODO: initial_pos and target_pos sampling should also be put here (scene-specific)

        else:
            # TODO: handcraft environments for more scenes
            assert False, 'model_id unknown'

        if self.arena in ['push_door', 'button_door']:
            self.door_axis_link_id = 1
            self.doors = []
            for scale, position, rotation in zip(door_scales, self.door_positions, self.door_rotations):
                door = InteractiveObj(
                    os.path.join(gibson2.assets_path, 'models', 'scene_components', 'realdoor.urdf'),
                    scale=scale)
                self.simulator.import_interactive_object(door, class_id=2)
                door.set_position_rotation(position, quatToXYZW(euler2quat(0, 0, rotation), 'wxyz'))
                self.doors.append(door)

            self.walls = []
            for wall_pose in wall_poses:
                wall = InteractiveObj(
                    os.path.join(gibson2.assets_path, 'models', 'scene_components', 'walls_quarter.urdf'),
                    scale=0.3)
                self.simulator.import_interactive_object(wall, class_id=3)
                wall.set_position_rotation(wall_pose[0], wall_pose[1])
                self.walls.append(wall)

            if self.arena == 'button_door':
                self.button_axis_link_id = 1
                self.button_threshold = -0.05
                self.button_reward = 5.0

                self.buttons = []
                for scale in button_scales:
                    button = InteractiveObj(
                        os.path.join(gibson2.assets_path, 'models', 'scene_components', 'eswitch', 'eswitch.urdf'),
                        scale=scale)
                    self.simulator.import_interactive_object(button, class_id=255)
                    self.buttons.append(button)

        elif self.arena == 'obstacles':
            self.obstacles = []
            for obstacle_pose in self.obstacle_poses:
                obstacle = BoxShape(pos=obstacle_pose, dim=[0.25, 0.25, 0.5], mass=10, color=[1, 0.64, 0, 1])
                self.simulator.import_interactive_object(obstacle, class_id=4)
                p.changeDynamics(obstacle.body_id, -1, lateralFriction=0.5)
                self.obstacles.append(obstacle)

    def prepare_motion_planner(self):
        self.robot_id = self.robots[0].robot_ids[0]
        self.mesh_id = self.scene.mesh_body_id
        self.map_size = self.scene.trav_map_original_size * self.scene.trav_map_default_resolution

        self.grid_resolution = 400
        self.occupancy_range = 8.0  # m
        robot_footprint_radius = 0.279
        self.robot_footprint_radius_in_map = int(robot_footprint_radius / self.occupancy_range * self.grid_resolution)

    def plan_base_motion(self, x, y, theta):
        half_size = self.map_size / 2.0
        # if self.mode == 'gui':
        #     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        path = plan_base_motion(self.robot_id, [x, y, theta], ((-half_size, -half_size), (half_size, half_size)),
                                obstacles=[self.mesh_id])
        # if self.mode == 'gui':
        #     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        return path

    def plan_base_motion_2d(self, x, y, theta):
        half_size = self.map_size / 2.0
        # if self.mode == 'gui':
        #     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)

        grid = self.get_local_occupancy_grid()
        path = plan_base_motion_2d(self.robot_id, [x, y, theta], ((-half_size, -half_size), (half_size, half_size)),
                                   map_2d=grid, occupancy_range=self.occupancy_range,
                                   grid_resolution=self.grid_resolution,
                                   robot_footprint_radius_in_map=self.robot_footprint_radius_in_map, obstacles=[])
        # if self.mode == 'gui':
        #     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        return path

    def global_to_local(self, pos, cur_pos, cur_rot):
        return rotate_vector_3d(pos - cur_pos, *cur_rot)

    def get_local_occupancy_grid(self):
        assert 'scan' in self.output
        assert self.config['robot'] in ['Turtlebot', 'Fetch']

        if self.config['robot'] == 'Turtlebot':
            # Hokuyo URG-04LX-UG01
            laser_linear_range = 5.6
            laser_angular_range = 240.0
            min_laser_dist = 0.05
            laser_link_name = 'scan_link'
        elif self.config['robot'] == 'Fetch':
            # SICK TiM571-2050101 Laser Range Finder
            laser_linear_range = 25.0
            laser_angular_range = 220.0
            min_laser_dist = 0.1
            laser_link_name = 'laser_link'

        laser_angular_half_range = laser_angular_range / 2.0
        laser_pose = self.robots[0].parts[laser_link_name].get_pose()
        base_pose = self.robots[0].parts['base_link'].get_pose()

        angle = np.arange(-laser_angular_half_range / 180 * np.pi,
                          laser_angular_half_range / 180 * np.pi,
                          laser_angular_range / 180.0 * np.pi / self.n_horizontal_rays)
        unit_vector_laser = np.array([[np.cos(ang), np.sin(ang), 0.0] for ang in angle])

        state = self.get_state()
        scan = state['scan']

        scan_laser = unit_vector_laser * (scan * (laser_linear_range - min_laser_dist) + min_laser_dist)

        laser_translation = laser_pose[:3]
        laser_rotation = quat2mat([laser_pose[6], laser_pose[3], laser_pose[4], laser_pose[5]])
        scan_world = laser_rotation.dot(scan_laser.T).T + laser_translation

        base_translation = base_pose[:3]
        base_rotation = quat2mat([base_pose[6], base_pose[3], base_pose[4], base_pose[5]])
        scan_local = base_rotation.T.dot((scan_world - base_translation).T).T
        scan_local = scan_local[:, :2]
        scan_local = np.concatenate([np.array([[0, 0]]), scan_local, np.array([[0, 0]])], axis=0)

        # flip y axis
        scan_local[:, 1] *= -1
        occupancy_grid = np.zeros((self.grid_resolution, self.grid_resolution)).astype(np.uint8)
        scan_local_in_map = scan_local / (self.occupancy_range / 2) * (self.grid_resolution / 2) + (
                self.grid_resolution / 2)
        scan_local_in_map = scan_local_in_map.reshape((1, -1, 1, 2)).astype(np.int32)
        cv2.fillPoly(occupancy_grid, scan_local_in_map, True, 1)
        cv2.circle(occupancy_grid, (self.grid_resolution // 2, self.grid_resolution // 2),
                   int(self.robot_footprint_radius_in_map), 1, -1)

        # cv2.imwrite('occupancy_grid.png', occupancy_grid)
        return occupancy_grid

    def get_additional_states(self):
        pos_noise = 0.0
        cur_pos = self.robots[0].get_position()
        cur_pos[:2] += np.random.normal(0, pos_noise, 2)

        rot_noise = 0.0 / 180.0 * np.pi
        cur_rot = self.robots[0].get_rpy()
        cur_rot = (cur_rot[0], cur_rot[1], cur_rot[2] + np.random.normal(0, rot_noise))

        target_pos_local = self.global_to_local(self.target_pos, cur_pos, cur_rot)
        # linear_velocity_local = rotate_vector_3d(self.robots[0].robot_body.velocity(), *cur_rot)[:2]
        # angular_velocity_local = rotate_vector_3d(self.robots[0].robot_body.angular_velocity(), *cur_rot)[2:3]

        gt_pos = self.robots[0].get_position()[:2]
        source = gt_pos
        target = self.target_pos[:2]
        _, geodesic_dist = self.scene.get_shortest_path(self.floor_num, source, target)
        # geodesic_dist = 0.0
        robot_z = self.robots[0].get_position()[2]
        if self.visualize_waypoints and self.mode == 'gui':
            for i in range(1000):
                self.waypoints_vis[i].set_position(pos=np.array([0.0, 0.0, 0.0]))
            for i in range(min(1000, self.shortest_path.shape[0])):
                self.waypoints_vis[i].set_position(pos=np.array([self.shortest_path[i][0],
                                                                 self.shortest_path[i][1],
                                                                 robot_z]))

        closest_idx = np.argmin(np.linalg.norm(cur_pos[:2] - self.shortest_path, axis=1))
        # approximate geodesic_dist to speed up training
        # geodesic_dist = np.sum(
        #     np.linalg.norm(self.shortest_path[closest_idx:-1] - self.shortest_path[closest_idx + 1:], axis=1)
        # )
        shortest_path = self.shortest_path[closest_idx:closest_idx + self.scene.num_waypoints]
        num_remaining_waypoints = self.scene.num_waypoints - shortest_path.shape[0]
        if num_remaining_waypoints > 0:
            remaining_waypoints = np.tile(self.target_pos[:2], (num_remaining_waypoints, 1))
            shortest_path = np.concatenate((shortest_path, remaining_waypoints), axis=0)

        shortest_path = np.concatenate((shortest_path, robot_z * np.ones((shortest_path.shape[0], 1))), axis=1)

        waypoints_local_xy = np.array([self.global_to_local(waypoint, cur_pos, cur_rot)[:2]
                                       for waypoint in shortest_path]).flatten()

        # # convert Cartesian space to radian space
        # for i in range(waypoints_local_xy.shape[0] // 2):
        #     vec = waypoints_local_xy[(i * 2):(i * 2 + 2)]
        #     norm = np.linalg.norm(vec)
        #     if norm == 0:
        #         continue
        #     dir = np.arctan2(vec[1], vec[0])
        #     waypoints_local_xy[i * 2] = dir
        #     waypoints_local_xy[i * 2 + 1] = norm
        #
        # norm = np.linalg.norm(target_pos_local[:2])
        # if norm != 0:
        #     dir = np.arctan2(target_pos_local[1], target_pos_local[0])
        #     target_pos_local[0] = dir
        #     target_pos_local[1] = norm

        additional_states = np.concatenate((waypoints_local_xy,
                                            target_pos_local[:2]))
        # linear_velocity_local,
        # angular_velocity_local))

        # cache results for reward calculation
        self.new_potential = geodesic_dist

        assert len(additional_states) == self.additional_states_dim, \
            'additional states dimension mismatch, {}, {}'.format(len(additional_states), self.additional_states_dim)

        return additional_states

    def get_state(self, collision_links=[]):
        state = super(MotionPlanningBaseArmEnv, self).get_state(collision_links)
        for modality in ['depth', 'pc']:
            if modality in state:
                img = state[modality]
                # width = img.shape[0]
                # height = int(width * (480.0 / 640.0))
                # half_diff = int((width - height) / 2)
                # img = img[half_diff:half_diff+height, :]
                if modality == 'depth':
                    high = 20.0
                    img[img > high] = high
                    img /= high
                state[modality] = img

        # cv2.imshow('depth', state['depth'])
        # cv2.imshow('scan', state['scan'])

        return state

    def get_potential(self):
        return self.new_potential

    def after_reset_agent(self):
        source = self.robots[0].get_position()[:2]
        target = self.target_pos[:2]
        shortest_path, geodesic_dist = self.scene.get_shortest_path(self.floor_num, source, target, entire_path=True)
        self.shortest_path = shortest_path
        self.new_potential = geodesic_dist

    def get_base_subgoal(self, action):
        """
        Convert action to base_subgoal
        :param action: policy output
        :return: base_subgoal_pos [x, y] in the world frame
        :return: base_subgoal_orn yaw in the world frame
        """
        # print('base')
        yaw = self.robots[0].get_rpy()[2]
        robot_pos = self.robots[0].get_position()
        base_subgoal_theta = (action[1] * 110.0) / 180.0 * np.pi  # [-110.0, 110.0]
        base_subgoal_theta += yaw
        base_subgoal_dist = (action[2] + 1)  # [0.0, 2.0]
        base_subgoal_pos = np.array([np.cos(base_subgoal_theta), np.sin(base_subgoal_theta)])
        base_subgoal_pos *= base_subgoal_dist
        base_subgoal_pos = np.append(base_subgoal_pos, 0.0)
        base_subgoal_pos += robot_pos
        base_subgoal_orn = action[3] * np.pi
        base_subgoal_orn += yaw

        # print('base_subgoal_pos', base_subgoal_pos)
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

        path = self.plan_base_motion_2d(base_subgoal_pos[0], base_subgoal_pos[1], base_subgoal_orn)
        if path is not None:
            # print('base mp success')
            if self.eval:
                for way_point in path:
                    set_base_values_with_z(self.robot_id, [way_point[0], way_point[1], way_point[2]],
                                           z=self.initial_height)
                    time.sleep(0.02)
            else:
                set_base_values_with_z(self.robot_id, [base_subgoal_pos[0], base_subgoal_pos[1], base_subgoal_orn],
                                       z=self.initial_height)

            return True
        else:
            # print('base mp failure')
            set_base_values_with_z(self.robot_id, original_pos, z=self.initial_height)
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
        # print('get_base_subgoal', time.time() - start)

        # start = time.time()
        subgoal_success = self.reach_base_subgoal(base_subgoal_pos, base_subgoal_orn)
        # print('reach_base_subgoal', time.time() - start)

        return subgoal_success

    def get_arm_subgoal(self, action):
        """
        Convert action to arm_subgoal
        :param action: policy output
        :return: arm_subgoal [x, y, z] in the world frame
        """
        state = self.get_state()
        points = state['pc']
        height, width = points.shape[0:2]

        arm_img_u = np.clip(int((action[4] + 1) / 2.0 * height), 0, height - 1)
        arm_img_v = np.clip(int((action[5] + 1) / 2.0 * width), 0, width - 1)

        point = points[arm_img_u, arm_img_v]
        camera_pose = (self.robots[0].parts['eyes'].get_pose())
        transform_mat = quat_pos_to_mat(pos=camera_pose[:3],
                                        quat=[camera_pose[6], camera_pose[3], camera_pose[4], camera_pose[5]])
        arm_subgoal = transform_mat.dot(np.array([-point[2], -point[0], point[1], 1]))[:3]
        self.arm_marker.set_position(arm_subgoal)
        return arm_subgoal

    def is_collision_free(self, body_a, link_a_list, body_b=None, link_b_list=None):
        """
        :param body_a: body id of body A
        :param link_a_list: link ids of body A that that of interest
        :param body_b: body id of body B (optional)
        :param link_b_list: link ids of body B that are of interest (optional)
        :return: whether the bodies and links of interest are collision-free
        """
        if body_b is None:
            for link_a in link_a_list:
                contact_pts = p.getContactPoints(bodyA=body_a, linkIndexA=link_a)
                if len(contact_pts) > 0:
                    return False
        elif link_b_list is None:
            for link_a in link_a_list:
                contact_pts = p.getContactPoints(bodyA=body_a, bodyB=body_b, linkIndexA=link_a)
                if len(contact_pts) > 0:
                    return False
        else:
            for link_a in link_a_list:
                for link_b in link_b_list:
                    contact_pts = p.getContactPoints(bodyA=body_a, bodyB=body_b, linkIndexA=link_a, linkIndexB=link_b)
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
        max_limits = [0., 0.] + get_max_limits(self.robot_id, self.arm_joint_ids)
        min_limits = [0., 0.] + get_min_limits(self.robot_id, self.arm_joint_ids)
        rest_position = [0., 0.] + list(get_joint_positions(self.robot_id, self.arm_joint_ids))
        joint_range = list(np.array(max_limits) - np.array(min_limits))
        joint_range = [item + 1 for item in joint_range]
        joint_damping = [0.1 for _ in joint_range]

        n_attempt = 0
        max_attempt = 50
        sample_fn = get_sample_fn(self.robot_id, self.arm_joint_ids)
        base_pose = get_base_values(self.robot_id)

        # find collision-free IK solution for arm_subgoal
        while n_attempt < max_attempt:
            set_joint_positions(self.robot_id, self.arm_joint_ids, sample_fn())
            arm_joint_positions = p.calculateInverseKinematics(self.robot_id,
                                                               self.robots[0].parts['gripper_link'].body_part_index,
                                                               arm_subgoal,
                                                               lowerLimits=min_limits,
                                                               upperLimits=max_limits,
                                                               jointRanges=joint_range,
                                                               restPoses=rest_position,
                                                               jointDamping=joint_damping,
                                                               solver=p.IK_DLS,
                                                               maxNumIterations=100)[2:10]
            set_joint_positions(self.robot_id, self.arm_joint_ids, arm_joint_positions)

            dist = l2_distance(self.robots[0].get_end_effector_position(), arm_subgoal)
            if dist > self.arm_subgoal_threshold:
                n_attempt += 1
                continue

            self.simulator_step()
            set_base_values_with_z(self.robot_id, base_pose, z=self.initial_height)
            self.reset_object_states()

            # arm should not have any collision
            collision_free = self.is_collision_free(body_a=self.robot_id,
                                                    link_a_list=self.arm_joint_ids)
            if not collision_free:
                n_attempt += 1
                continue

            # gripper should not have any self-collision
            collision_free = self.is_collision_free(body_a=self.robot_id,
                                                    link_a_list=[self.robots[0].parts['gripper_link'].body_part_index],
                                                    body_b=self.robot_id)
            if not collision_free:
                n_attempt += 1
                continue

            return arm_joint_positions

        return

    def reset_obstacles_z(self):
        """
        Make all obstacles perpendicular to the ground
        """
        for obstacle in self.obstacles:
            obstacle_pose = get_base_values(obstacle.body_id)
            set_base_values_with_z(obstacle.body_id, obstacle_pose, 0.6)

    def reach_arm_subgoal(self, arm_joint_positions):
        """
        Attempt to reach arm arm_joint_positions and return success / failure
        If failed, reset the arm to its original pose
        :param arm_joint_positions
        :return: whether arm_joint_positions is achieved
        """
        set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)

        if arm_joint_positions is None:
            return False

        arm_path = plan_joint_motion(self.robot_id,
                                     self.arm_joint_ids,
                                     arm_joint_positions,
                                     disabled_collisions=set(),
                                     self_collisions=False)
        if arm_path is not None:
            if self.eval:
                for joint_way_point in arm_path:
                    set_joint_positions(self.robot_id, self.arm_joint_ids, joint_way_point)
                    time.sleep(0.02)  # animation
            else:
                set_joint_positions(self.robot_id, self.arm_joint_ids, arm_joint_positions)
            return True
        else:
            set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
            return False

    def stash_object_states(self):
        if self.arena in ['push_door', 'button_door']:
            for i, door in enumerate(self.doors):
                self.door_angles[i] = p.getJointState(door.body_id, self.door_axis_link_id)[0]
            if self.arena == 'button_door':
                for i, button in enumerate(self.buttons):
                    self.button_states[i] = p.getJointState(button.body_id, self.button_axis_link_id)[0]
        elif self.arena == 'obstacles':
            for i, obstacle in enumerate(self.obstacles):
                self.obstacle_states[i] = p.getBasePositionAndOrientation(obstacle.body_id)

    def reset_object_states(self):
        """
        Remove any accumulated velocities or forces of objects resulting from arm motion planner
        """
        if self.arena in ['push_door', 'button_door']:
            for door, door_angle in zip(self.doors, self.door_angles):
                p.resetJointState(door.body_id, self.door_axis_link_id,
                                  targetValue=door_angle, targetVelocity=0.0)
            # for wall in self.walls:
            #     p.resetBaseVelocity(wall.body_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            if self.arena == 'button_door':
                for button, button_state in zip(self.buttons, self.button_states):
                    p.resetJointState(button.body_id, self.button_axis_link_id,
                                      targetValue=button_state, targetVelocity=0.0)
        elif self.arena == 'obstacles':
            for obstacle, obstacle_state in zip(self.obstacles, self.obstacle_states):
                p.resetBasePositionAndOrientation(obstacle.body_id, *obstacle_state)

    def interact(self, action, arm_subgoal):
        """
        Move the arm according to push_vector and physically simulate the interaction
        :param action: policy output
        :param arm_subgoal: starting location of the interaction
        :return: None
        """
        push_vector_local = np.array([action[6], action[7]])  # [-1.0, 1.0]
        push_vector = rotate_vector_2d(push_vector_local, -self.robots[0].get_rpy()[2])
        push_vector = np.append(push_vector, 0.0)

        # push_vector = np.array([-0.5, 0.0, 0.0])

        max_limits = [0., 0.] + get_max_limits(self.robot_id, self.arm_joint_ids)
        min_limits = [0., 0.] + get_min_limits(self.robot_id, self.arm_joint_ids)
        rest_position = [0., 0.] + list(get_joint_positions(self.robot_id, self.arm_joint_ids))
        joint_range = list(np.array(max_limits) - np.array(min_limits))
        joint_range = [item + 1 for item in joint_range]
        joint_damping = [0.1 for _ in joint_range]

        base_pose = get_base_values(self.robot_id)

        # self.simulator.set_timestep(0.002)
        for i in range(100):
            push_goal = np.array(arm_subgoal) + push_vector * i / 100.0

            joint_positions = p.calculateInverseKinematics(self.robot_id,
                                                           self.robots[0].parts['gripper_link'].body_part_index,
                                                           push_goal,
                                                           lowerLimits=min_limits,
                                                           upperLimits=max_limits,
                                                           jointRanges=joint_range,
                                                           restPoses=rest_position,
                                                           jointDamping=joint_damping,
                                                           solver=p.IK_DLS,
                                                           maxNumIterations=100)[2:10]

            # set_joint_positions(self.robot_id, self.arm_joint_ids, joint_positions)
            control_joints(self.robot_id, self.arm_joint_ids, joint_positions)
            self.simulator_step()
            set_base_values_with_z(self.robot_id, base_pose, z=self.initial_height)

            if self.arena == 'obstacles':
                self.reset_obstacles_z()

            if self.eval:
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
        # print('get_arm_subgoal', time.time() - start)

        # start = time.time()
        # print(p.getNumBodies())
        # state_id = p.saveState()
        # print('saveState', time.time() - start)

        self.stash_object_states()

        # start = time.time()
        arm_joint_positions = self.get_arm_joint_positions(arm_subgoal)
        # print('get_arm_joint_positions', time.time() - start)

        # start = time.time()
        subgoal_success = self.reach_arm_subgoal(arm_joint_positions)
        # print('reach_arm_subgoal', time.time() - start)

        # start = time.time()
        # p.restoreState(stateId=state_id)
        # print('restoreState', time.time() - start)

        # start = time.time()
        self.reset_object_states()
        # print('reset_object_velocities', time.time() - start)

        if subgoal_success:
            # set_joint_positions(self.robot_id, self.arm_joint_ids, arm_joint_positions)

            # start = time.time()
            self.interact(action, arm_subgoal)
            # print('interact', time.time() - start)

        return subgoal_success

    def step(self, action):
        # print('-' * 30)
        # embed()
        # action[0] = base_or_arm
        # action[1] = base_subgoal_theta
        # action[2] = base_subgoal_dist
        # action[3] = base_orn
        # action[4] = arm_img_u
        # action[5] = arm_img_v
        # action[6] = arm_push_vector_x
        # action[7] = arm_push_vector_y

        self.current_step += 1
        use_base = action[0] > 0.0
        if use_base:
            subgoal_success = self.move_base(action)
        else:
            subgoal_success = self.move_arm(action)

        # print('subgoal success', subgoal_success)

        return self.compute_next_step(action, use_base, subgoal_success)

    def compute_next_step(self, action, use_base, subgoal_success):
        self.simulator.sync()

        if use_base:
            # trigger re-computation of geodesic distance for get_reward
            state = self.get_state()

        info = {}
        if subgoal_success:
            reward, info = self.get_reward([], action, info)
        else:
            # failed subgoal penalty
            reward = self.failed_subgoal_penalty
        done, info = self.get_termination([], info)

        if self.arena == 'button_door':
            button_state = p.getJointState(self.buttons[self.door_idx].body_id, self.button_axis_link_id)[0]
            if not self.button_pressed and button_state < self.button_threshold:
                print("OPEN DOOR")
                self.button_pressed = True
                self.doors[self.door_idx].set_position([100.0, 100.0, 0.0])
                reward += self.button_reward
        elif self.arena == 'push_door':
            new_door_angle = p.getJointState(self.doors[self.door_idx].body_id, self.door_axis_link_id)[0]
            door_angle_diff = new_door_angle - self.door_angles[self.door_idx]
            reward += door_angle_diff
            self.door_angles[self.door_idx] = new_door_angle

        if not use_base:
            set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
            state = self.get_state()

        if done and self.automatic_reset:
            state = self.reset()

        del state['pc']

        # print('reward', reward)
        # time.sleep(3)

        return state, reward, done, info

    def reset_initial_and_target_pos(self):
        if self.arena in ['button_door', 'push_door', 'obstacles']:
            floor_height = self.scene.get_floor_height(self.floor_num)
            self.initial_height = floor_height + self.random_init_z_offset
            self.initial_pos = np.array([1.2, 0.0, floor_height])
            self.robots[0].set_position(pos=[self.initial_pos[0],
                                             self.initial_pos[1],
                                             self.initial_height])
            self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(0, 0, np.pi), 'wxyz'))

            if self.arena in ['button_door', 'push_door']:
                self.door_idx = np.random.randint(0, len(self.doors))
                door_target_pos = self.door_target_pos[self.door_idx]
                self.target_pos = np.array([
                    np.random.uniform(door_target_pos[0][0], door_target_pos[0][1]),
                    np.random.uniform(door_target_pos[1][0], door_target_pos[1][1]),
                    floor_height
                ])
            else:
                self.target_pos = np.array([-5.0, 0.0, floor_height])
        else:
            super(MotionPlanningBaseArmEnv, self).reset_initial_and_target_pos()

    def before_reset_agent(self):
        if self.arena in ['push_door', 'button_door']:
            self.door_angles = np.zeros(len(self.doors))
            for door, angle, pos, orn in zip(self.doors, self.door_angles, self.door_positions, self.door_rotations):
                p.resetJointState(door.body_id, self.door_axis_link_id, targetValue=angle, targetVelocity=0.0)
                door.set_position_rotation(pos, quatToXYZW(euler2quat(0, 0, orn), 'wxyz'))
            if self.arena == 'button_door':
                self.button_pressed = False
                self.button_states = np.zeros(len(self.buttons))
                for button, button_pos_range, button_rotation, button_state in \
                        zip(self.buttons, self.button_positions, self.button_rotations, self.button_states):
                    button_pos = np.array([
                        np.random.uniform(button_pos_range[0][0], button_pos_range[0][1]),
                        np.random.uniform(button_pos_range[1][0], button_pos_range[1][1]),
                        1.5
                    ])
                    button.set_position_rotation(button_pos, quatToXYZW(euler2quat(0, 0, button_rotation), 'wxyz'))
                    p.resetJointState(button.body_id, self.button_axis_link_id,
                                      targetValue=button_state, targetVelocity=0.0)
        elif self.arena == 'obstacles':
            self.obstacle_states = [None] * len(self.obstacles)
            for obstacle, obstacle_pose in zip(self.obstacles, self.obstacle_poses):
                set_base_values_with_z(obstacle.body_id, [obstacle_pose[0], obstacle_pose[1], 0], 0.6)

    def reset(self):
        state = super(MotionPlanningBaseArmEnv, self).reset()
        del state['pc']
        return state


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

    parser.add_argument('--arena',
                        '-a',
                        choices=['button_door', 'push_door', 'obstacles'],
                        default='push_door',
                        help='which arena to train or test (default: push_door)')

    args = parser.parse_args()

    nav_env = MotionPlanningBaseArmEnv(config_file=args.config,
                                       mode=args.mode,
                                       action_timestep=1 / 500.0,
                                       physics_timestep=1 / 500.0,
                                       eval=args.mode == 'gui',
                                       arena=args.arena,
                                       )

    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        state = nav_env.reset()
        for i in range(150):
            # print('Step: {}'.format(i))
            action = nav_env.action_space.sample()
            state, reward, done, info = nav_env.step(action)
            # embed()
            # print('Reward:', reward)
            # time.sleep(0.05)
            # nav_env.step()
            # for step in range(50):  # 500 steps, 50s world time
            #    action = nav_env.action_space.sample()
            #    state, reward, done, _ = nav_env.step(action)
            #    # print('reward', reward)
            if done:
                print('Episode finished after {} timesteps'.format(i + 1))
                break
        print(time.time() - start)
    nav_env.clean()
