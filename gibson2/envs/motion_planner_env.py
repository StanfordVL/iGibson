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
from gibson2.core.render.utils import quat_pos_to_mat
from gibson2.external.pybullet_tools.utils import set_base_values, joint_from_name, set_joint_position, \
    set_joint_positions, add_data_path, connect, plan_base_motion, plan_joint_motion, enable_gravity, \
    joint_controller, dump_body, load_model, joints_from_names, user_input, disconnect, get_joint_positions, \
    get_link_pose, link_from_name, HideOutput, get_pose, wait_for_user, dump_world, plan_nonholonomic_motion, \
    set_point, create_box, stable_z, control_joints, get_max_limits, get_min_limits, get_base_values, \
    plan_base_motion_2d, get_sample_fn, add_p2p_constraint, remove_constraint


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
        self.action_space = gym.spaces.Box(shape=(6,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.prepare_motion_planner()

        # visualization
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
        self.arm_subgoal_threshold = 0.05

        self.push_dist_threshold = 0.01

        self.failed_subgoal_penalty = -0.0

        if self.arena == 'button':
            self.button_marker = VisualMarker(visual_shape=p.GEOM_SPHERE,
                                              rgba_color=[0, 1, 0, 1],
                                              radius=0.3)
            self.simulator.import_object(self.button_marker, class_id=255)
            self.button_threshold = 0.5
            self.button_reward = 5.0

            self.door = InteractiveObj(
                os.path.join(gibson2.assets_path, 'models', 'scene_components', 'realdoor.urdf'),
                scale=4.0)
            self.simulator.import_interactive_object(self.door, class_id=2)

            self.wall_poses = [
                [[0, -2.0, 1], [0, 0, 0, 1]],
            ]
            self.walls = []
            for wall_pose in self.wall_poses:
                wall = InteractiveObj(os.path.join(gibson2.assets_path, 'models', 'scene_components', 'walls.urdf'),
                                      scale=1)
                self.simulator.import_interactive_object(wall, class_id=3)
                wall.set_position_rotation(wall_pose[0], wall_pose[1])
                self.simulator.sync()
                self.walls += [wall]
        elif self.arena == 'push_door':
            self.door = InteractiveObj(
                os.path.join(gibson2.assets_path, 'models', 'scene_components', 'realdoor.urdf'),
                scale=1.0)
            self.simulator.import_interactive_object(self.door, class_id=2)
            self.door.set_position_rotation([-3.5, 0, 0.0], quatToXYZW(euler2quat(0, 0, np.pi / 2.0), 'wxyz'))
            self.door_axis_link_id = 1
            # for i in range(p.getNumJoints(self.door.body_id)):
            #    for j in range(p.getNumJoints(self.robot_id)):
            #        #if j != self.robots[0].parts['gripper_link'].body_part_index:
            #        p.setCollisionFilterPair(self.door.body_id, self.robot_id, i, j, 0) # disable collision for robot and door

            self.walls = []
            wall = BoxShape([-3.5, 1, 0.7], [0.2, 0.35, 0.7], visual_only=True)
            self.simulator.import_interactive_object(wall, class_id=3)
            self.walls += [wall]
            wall = BoxShape([-3.5, -1, 0.7], [0.2, 0.45, 0.7], visual_only=True)
            self.simulator.import_interactive_object(wall, class_id=3)
            self.walls += [wall]

            self.simulator.sync()

    def prepare_motion_planner(self):
        self.robot_id = self.robots[0].robot_ids[0]
        self.mesh_id = self.scene.mesh_body_id
        self.map_size = self.scene.trav_map_original_size * self.scene.trav_map_default_resolution

        self.grid_resolution = 400
        self.occupancy_range = 8.0  # m
        self.robot_footprint_radius = 0.279
        self.robot_footprint_radius_in_map = int(
            self.robot_footprint_radius / self.occupancy_range * self.grid_resolution)

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
        # assumes it has "scan" state
        # assumes it is either Turtlebot or fetch
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
        # embed()

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
        # embed()
        # assert False
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

        # # convert Cartesian space to radian space START
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
        # # convert Cartesian space to radian space END

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

    def step(self, action):
        # print('-' * 30)
        # action[0] = base_or_arm
        # action[1] = base_subgoal_theta
        # action[2] = base_subgoal_dist
        # action[3] = base_orn
        # action[4] = arm_img_u
        # action[5] = arm_img_v

        self.current_step += 1
        use_base = action[0] > 0.0
        if use_base:
            # print('base')
            # use base
            yaw = self.robots[0].get_rpy()[2]
            robot_pos = self.robots[0].get_position()
            base_subgoal_theta = (action[1] * 110.0) / 180.0 * np.pi  # [-110.0, 110.0]
            base_subgoal_theta += yaw
            base_subgoal_dist = (action[2] + 1)  # [0.0, 2.0]
            base_subgoal_pos = np.array([np.cos(base_subgoal_theta), np.sin(base_subgoal_theta)])
            base_subgoal_pos *= base_subgoal_dist
            base_subgoal_pos = np.append(base_subgoal_pos, 0.0)
            base_subgoal_pos += robot_pos
            # print('base_subgoal_pos', base_subgoal_pos)

            self.base_marker.set_position(base_subgoal_pos)

            base_subgoal_orn = action[3] * np.pi
            base_subgoal_orn += yaw

            original_pos = get_base_values(self.robot_id)

            path = self.plan_base_motion_2d(base_subgoal_pos[0], base_subgoal_pos[1], base_subgoal_orn)
            subgoal_success = path is not None
            if subgoal_success:
                # print('base mp success')
                if self.eval:
                    for way_point in path:
                        set_base_values(self.robot_id, [way_point[0], way_point[1], way_point[2]])
                        time.sleep(0.02)
                else:
                    set_base_values(self.robot_id, [base_subgoal_pos[0], base_subgoal_pos[1], base_subgoal_orn])
            else:
                # print('base mp failure')
                set_base_values(self.robot_id, original_pos)

            # is_base_subgoal_valid = self.scene.has_node(self.floor_num, base_subgoal_pos[:2])
            # if self.arena == 'button':
            #     x_min = -6.0 if self.door_open else -3.0
            #     x_max = 2.0
            #     y_min, y_max = -2.0, 2.0
            #     is_valid = x_min <= base_subgoal_pos[0] <= x_max and y_min <= base_subgoal_pos[1] <= y_max
            #     is_base_subgoal_valid = is_base_subgoal_valid and is_valid
            # if is_base_subgoal_valid:
            #     if self.eval:
            #         path = self.plan_base_motion_2d(base_subgoal_pos[0], base_subgoal_pos[1], base_subgoal_orn)
            #         if path is not None:
            #             print('base mp success')
            #             for way_point in path:
            #                 set_base_values(self.robot_id, [way_point[0], way_point[1], way_point[2]])
            #                 time.sleep(0.05)
            #         else:
            #             print('base mp failure')
            #     else:
            #         set_base_values(self.robot_id, [base_subgoal_pos[0], base_subgoal_pos[1], base_subgoal_orn])
            #     # set_base_values(self.robot_id, [base_subgoal_pos[0], base_subgoal_pos[1], base_subgoal_orn])
            #     # print('subgoal succeed')
            # else:
            #     # print('subgoal fail')
            #     subgoal_success = False
        else:
            # print('arm')
            # use arm
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

            arm_joints = joints_from_names(self.robot_id,
                                           ['torso_lift_joint',
                                            'shoulder_pan_joint',
                                            'shoulder_lift_joint',
                                            'upperarm_roll_joint',
                                            'elbow_flex_joint',
                                            'forearm_roll_joint',
                                            'wrist_flex_joint',
                                            'wrist_roll_joint'])
            max_limits = [0., 0.] + get_max_limits(self.robot_id, arm_joints)
            min_limits = [0., 0.] + get_min_limits(self.robot_id, arm_joints)
            rest_position = [0., 0.] + list(get_joint_positions(self.robot_id, arm_joints))
            joint_range = list(np.array(max_limits) - np.array(min_limits))
            joint_range = [item + 1 for item in joint_range]
            joint_damping = [0.1 for _ in joint_range]

            n_attempt = 0
            max_attempt = 2000
            sample_fn = get_sample_fn(self.robot_id, arm_joints)

            while n_attempt < max_attempt:  # find self-collision-free ik solution
                set_joint_positions(self.robot_id, arm_joints, sample_fn())
                subgoal_joint_positions = p.calculateInverseKinematics(self.robot_id,
                                                                       self.robots[0].parts[
                                                                           'gripper_link'].body_part_index,
                                                                       arm_subgoal,
                                                                       lowerLimits=min_limits,
                                                                       upperLimits=max_limits,
                                                                       jointRanges=joint_range,
                                                                       restPoses=rest_position,
                                                                       jointDamping=joint_damping,
                                                                       solver=p.IK_DLS,
                                                                       maxNumIterations=100)[2:10]
                set_joint_positions(self.robot_id, arm_joints, subgoal_joint_positions)

                dist = l2_distance(self.robots[0].get_end_effector_position(), arm_subgoal)
                if dist < self.arm_subgoal_threshold:
                    # print('arm_subgoal_dist', dist)
                    collision_free = True
                    num_joints = p.getNumJoints(self.robot_id)
                    # self collision
                    for arm_link in arm_joints:
                        for other_link in range(num_joints):
                            contact_pts = p.getContactPoints(self.robot_id, self.robot_id, arm_link, other_link)
                            # print(contact_pts)
                            if len(contact_pts) > 0:
                                collision_free = False
                                break
                        if not collision_free:
                            break

                    # # arm collision with door
                    # if self.arena == 'push_door':
                    #     for arm_link in arm_joints:
                    #         for door_link in range(p.getNumJoints(self.door.body_id)):
                    #             if arm_link != self.robots[0].parts['gripper_link'].body_part_index:
                    #                 contact_pts = p.getContactPoints(self.robot_id, self.door.body_id, arm_link,
                    #                                                  door_link)
                    #                 if len(contact_pts) > 0:
                    #                     print(arm_link, 'in collision with door')
                    #                     collision_free = False
                    #                     break
                    #         if not collision_free:
                    #             break

                    if collision_free:
                        break
                n_attempt += 1

            ik_success = n_attempt != max_attempt

            if ik_success:
                set_joint_positions(self.robot_id, arm_joints, self.arm_default_joint_positions)
                arm_path = plan_joint_motion(self.robot_id, arm_joints, subgoal_joint_positions,
                                             disabled_collisions=set(),
                                             self_collisions=False)
                subgoal_success = arm_path is not None
            else:
                subgoal_success = False

            set_joint_positions(self.robot_id, arm_joints, self.arm_default_joint_positions)

            # print('ik_success', ik_success)
            # print('arm_mp_success', subgoal_success)

            if subgoal_success:
                if self.eval:
                    for joint_way_point in arm_path:
                        set_joint_positions(self.robot_id, arm_joints, joint_way_point)
                        time.sleep(0.02)  # animation
                else:
                    set_joint_positions(self.robot_id, arm_joints, subgoal_joint_positions)

                ## push
                # TODO: figure out push dist threshold (i.e. can push object x centimeters away from the gripper)
                # TODO: whitelist object ids that can be pushed

                # find the closest object
                focus = None
                points = []
                pushable_obj_ids = [self.door.body_id]
                for i in pushable_obj_ids:
                    points.extend(
                        p.getClosestPoints(self.robot_id, i, distance=self.push_dist_threshold,
                                           linkIndexA=self.robots[0].parts['gripper_link'].body_part_index))
                dist = 1e4
                for point in points:
                    if point[8] < dist:
                        dist = point[8]
                        # if not focus is None and not (focus[2] == point[2] and focus[4] == point[4]):
                        # p.changeVisualShape(objectUniqueId=focus[2], linkIndex=focus[4], rgbaColor=[1, 1, 1, 1])
                        focus = point
                        # p.changeVisualShape(objectUniqueId=focus[2], linkIndex=focus[4], rgbaColor=[1, 0, 0, 1])

                # print(focus)

                # if focus is not None:
                #     c = add_p2p_constraint(focus[2],
                #                            focus[4],
                #                            self.robot_id,
                #                            self.robots[0].parts['gripper_link'].body_part_index,
                #                            max_force=500)

                push_vector = np.array([-0.5, 0.0, 0])

                base_pose = get_base_values(self.robot_id)

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
                    set_base_values(self.robot_id, base_pose)

                    set_joint_positions(self.robot_id, arm_joints, joint_positions)
                    self.simulator.set_timestep(0.002)
                    self.simulator_step()
                    self.simulator.set_timestep(1e-8)
                    if self.eval:
                        time.sleep(0.02)  # for visualization
                # if focus is not None:
                #    remove_constraint(c)

                set_base_values(self.robot_id, base_pose)

        ###### reward computation ######
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

        if self.arena == 'button':
            dist = l2_distance(self.robots[0].get_end_effector_position(), self.button_marker_pos)
            # print('button_marker_dist', dist)
            if not self.door_open and dist < self.button_threshold:
                # touch the button -> remove the button and the door
                print("OPEN DOOR")
                self.door_open = True
                self.button_marker.set_position([100.0, 100.0, 0.0])
                self.door.set_position([100.0, 100.0, 0.0])

                reward += self.button_reward

        if not use_base:
            # arm reset
            set_joint_positions(self.robot_id, arm_joints, self.arm_default_joint_positions)
            # need to call get_state again after arm reset (camera height could be different if torso moves)
            # TODO: should only call get_state once or twice (e.g. disable torso movement, or cache get_state result)
            state = self.get_state()

        if done and self.automatic_reset:
            state = self.reset()
        del state['pc']

        if use_base:
            info['start_conf'] = original_pos
            info['path'] = path
        # print('reward', reward)
        # time.sleep(3)

        self.simulator.sync()

        return state, reward, done, info

    def reset_initial_and_target_pos(self):
        if self.arena in ['button', 'push_door']:
            floor_height = self.scene.get_floor_height(self.floor_num)
            self.initial_pos = np.array([-3.0, 0.0, floor_height])
            self.target_pos = np.array([-5.0, 0.0, floor_height])
            self.robots[0].set_position(pos=[self.initial_pos[0],
                                             self.initial_pos[1],
                                             self.initial_pos[2] + self.random_init_z_offset])
            self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(0, 0, np.pi), 'wxyz'))

            if self.arena == 'button':
                self.button_marker_pos = [
                    np.random.uniform(-3.0, -0.5),
                    np.random.uniform(-1.25, 1.25),
                    1.5
                ]
                self.button_marker.set_position(self.button_marker_pos)
                self.door.set_position_rotation([-3.5, 0, 0.0], quatToXYZW(euler2quat(0, 0, np.pi / 2.0), 'wxyz'))
            elif self.arena == 'push_door':
                p.resetJointState(self.door.body_id, self.door_axis_link_id, targetValue=0.0, targetVelocity=0.0)

        else:
            super(MotionPlanningBaseArmEnv, self).reset_initial_and_target_pos()

    def reset(self):
        state = super(MotionPlanningBaseArmEnv, self).reset()
        del state['pc']
        # embed()
        if self.arena == 'button':
            self.door_open = False

        self.simulator.sync()
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

    args = parser.parse_args()

    nav_env = MotionPlanningBaseArmEnv(config_file=args.config,
                                       mode=args.mode,
                                       action_timestep=1.0 / 1000000.0,
                                       physics_timestep=1.0 / 1000000.0,
                                       eval=args.mode == 'gui',
                                       arena='push_door',
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
