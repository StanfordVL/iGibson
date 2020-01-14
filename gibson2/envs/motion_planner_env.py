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
    set_point, create_box, stable_z, control_joints, get_max_limits, get_min_limits, get_base_values


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

        self.mp_loaded = False
        # override some parameters:
        self.max_step = 20
        self.planner_step = 0
        self.action_space = gym.spaces.Box(shape=(3,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

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
        path = plan_base_motion(
            self.robot_id,
            [x, y, theta],
            ((-half_size, -half_size), (half_size, half_size)),
            obstacles=[self.mesh_id]
        )
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
            bq = path[-1]
            # for bq in path:
            set_base_values(self.robot_id, [bq[0], bq[1], bq[2]])
            state, _, done, info = super(MotionPlanningEnv, self).step([0, 0])
            # time.sleep(0.05)
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
                 ):
        super(MotionPlanningBaseArmEnv, self).__init__(config_file,
                                                       model_id=model_id,
                                                       mode=mode,
                                                       action_timestep=action_timestep,
                                                       physics_timestep=physics_timestep,
                                                       automatic_reset=automatic_reset,
                                                       random_height=False,
                                                       device_idx=device_idx)
        resolution = self.config.get('resolution', 64)
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
        # action[0] = arm_img_u
        # action[1] = arm_img_v
        # action[2] = base_dx
        # action[3] = base_dy
        # action[4] = base_dtheta
        self.action_space = gym.spaces.Box(shape=(5,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.prepare_motion_planner()

    def prepare_motion_planner(self):
        self.robot_id = self.robots[0].robot_ids[0]
        self.mesh_id = self.scene.mesh_body_id
        self.map_size = self.scene.trav_map_original_size * self.scene.trav_map_default_resolution
        # print(self.robot_id, self.mesh_id, self.map_size)
        self.base_marker = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                        rgba_color=[1, 0, 0, 1],
                                        radius=0.1,
                                        length=0.1,
                                        initial_offset=[0, 0, 0.1 / 2.0])
        self.base_marker.load()

        self.arm_marker = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                       rgba_color=[1, 1, 0, 1],
                                       radius=0.1,
                                       length=0.1,
                                       initial_offset=[0, 0, 0.1 / 2.0])
        self.arm_marker.load()

    def plan_base_motion(self, x, y, theta):
        half_size = self.map_size / 2.0
        path = plan_base_motion(self.robot_id, [x, y, theta], ((-half_size, -half_size), (half_size, half_size)),
                                obstacles=[self.mesh_id])
        return path

    def global_to_local(self, pos, cur_pos, cur_rot):
        return rotate_vector_3d(pos - cur_pos, *cur_rot)

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
        additional_states = np.concatenate((waypoints_local_xy,
                                            target_pos_local))
                                            # linear_velocity_local,
                                            # angular_velocity_local))
        # cache results for reward calculation
        self.new_potential = geodesic_dist
        assert len(additional_states) == self.additional_states_dim, 'additional states dimension mismatch'
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
                    high = 25.0
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
        self.current_step += 1

        state = self.get_state()
        points = state['pc']
        height, width = points.shape[0:2]

        arm_img_u = int((action[0] + 1) / 2.0 * height)
        arm_img_v = int((action[1] + 1) / 2.0 * width)

        # original_pos = get_base_values(self.robot_id)
        point = points[arm_img_u, arm_img_v]
        camera_pose = (self.robots[0].parts['eyes'].get_pose())
        transform_mat = quat_pos_to_mat(pos=camera_pose[:3],
                                        quat=[camera_pose[6], camera_pose[3], camera_pose[4], camera_pose[5]])
        arm_subgoal = transform_mat.dot(np.array([-point[2], -point[0], point[1], 1]))[:3]

        dx = action[2] * 0.5
        dy = action[3] * 0.5
        yaw = self.robots[0].get_rpy()[2]
        dtheta = action[4] * np.pi
        base_subgoal_orn = dtheta + yaw

        base_subgoal_pos = np.array([arm_subgoal[0] + dx, arm_subgoal[1] + dy, self.robots[0].get_position()[2]])
        self.arm_marker.set_position(arm_subgoal)
        self.base_marker.set_position(base_subgoal_pos)

        # embed()
        # print('before plan_base_motion')
        # path = self.plan_base_motion(base_subgoal_pos[0], base_subgoal_pos[1], base_subgoal_orn)
        # print('after plan_base_motion')
        # embed()
        # print('base mp finished -------------------')

        is_base_subgoal_valid = self.scene.has_node(self.floor_num, base_subgoal_pos[:2])
        if is_base_subgoal_valid:
        # if path is not None:
        #     print('subgoal succeed')
        #     last_step = path[-1]
            set_base_values(self.robot_id, [base_subgoal_pos[0], base_subgoal_pos[1], base_subgoal_orn])
            # state, _, done, info = super(MotionPlanningBaseArmEnv, self).step(None)
            # self.get_additional_states()
        else:
            # print('subgoal fail')
            # set_base_values(self.robot_id, original_pos)
            reward = -0.02
            # print('reward', reward)
            # time.sleep(3)
            info = {}
            done, info = self.get_termination([], info)

            if done and self.automatic_reset:
                info['last_observation'] = state
                state = self.reset()

            del state['pc']
            return state, reward, done, info

        arm_joints = joints_from_names(self.robot_id,
                                       ['torso_lift_joint',
                                        'shoulder_pan_joint',
                                        'shoulder_lift_joint',
                                        'upperarm_roll_joint',
                                        'elbow_flex_joint',
                                        'forearm_roll_joint',
                                        'wrist_flex_joint',
                                        'wrist_roll_joint'])
        max_limits = [0., 0.] + get_max_limits(self.robot_id, arm_joints) + [0.05, 0.05]
        min_limits = [0., 0.] + get_min_limits(self.robot_id, arm_joints) + [0., 0.]
        rest_position = [0., 0.] + list(get_joint_positions(self.robot_id, arm_joints)) + [0.04, 0.04]
        joint_range = list(np.array(max_limits) - np.array(min_limits))
        joint_range = [item + 1 for item in joint_range]
        joint_damping = [0.1 for _ in joint_range]
        # print('before arm IK')
        joint_positions = p.calculateInverseKinematics(self.robot_id,
                                                       self.robots[0].parts['gripper_link'].body_part_index,
                                                       arm_subgoal,
                                                       lowerLimits=min_limits,
                                                       upperLimits=max_limits,
                                                       jointRanges=joint_range,
                                                       restPoses=rest_position,
                                                       jointDamping=joint_damping,
                                                       solver=p.IK_DLS,
                                                       maxNumIterations=100)[2:10]
        # print('after arm IK')
        # print(joint_positions)
        set_joint_positions(self.robot_id, arm_joints, joint_positions)
        # dist = l2_distance(arm_subgoal, self.get_position_of_interest())
        # print(dist)

        # trigger re-computation of geodesic distance for get_reward
        self.get_state()

        info = {}
        reward, info = self.get_reward([], action, info)
        done, info = self.get_termination([], info)

        # arm reset
        arm_default_joint_positions = (0.38548146667743244, 1.1522793897208579,
                                       1.2576467971105596, -0.312703569911879,
                                       1.7404867100093226, -0.0962895617312548,
                                       -1.4418232619629425, -1.6780152866247762)
        set_joint_positions(self.robot_id, arm_joints, arm_default_joint_positions)

        # need to call get_state again after arm reset (camera height could be different if torso moves)
        # TODO: should only call get_state once or twice (e.g. disable torso movement, or cache get_state result)
        state = self.get_state()
        if done and self.automatic_reset:
            state = self.reset()
        del state['pc']
        # print('reward', reward)
        # time.sleep(3)

        return state, reward, done, info

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

    args = parser.parse_args()

    nav_env = MotionPlanningBaseArmEnv(config_file=args.config,
                                       mode=args.mode,
                                       action_timestep=1.0 / 1000000.0,
                                       physics_timestep=1.0 / 1000000.0)

    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        state = nav_env.reset()
        for i in range(150):
            # print('Step: {}'.format(i))
            action = nav_env.action_space.sample()
            state, reward, done, info = nav_env.step(action)
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
