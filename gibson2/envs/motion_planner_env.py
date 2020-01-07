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
                                           low = -1.0,
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

    def plan_base_motion(self, x,y,theta):
        half_size = self.map_size / 2.0
        path = plan_base_motion(self.robot_id, [x,y,theta], ((-half_size, -half_size), (half_size, half_size)), obstacles=[self.mesh_id])
        return path

    def step(self, pt):
        # point = [x,y]
        x = int((pt[0]+1)/2.0 * 150)
        y = int((pt[1]+1)/2.0 * 128)
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
            #for bq in path:
            set_base_values(self.robot_id, [bq[0], bq[1], bq[2]])
            state, _, done, info = super(MotionPlanningEnv, self).step([0, 0])
            #time.sleep(0.05)
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
        #print(info)
        #if info['success']:
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

        self.mp_loaded = False
        # override some parameters:
        self.max_step = 20
        self.planner_step = 0
        self.action_space = gym.spaces.Box(shape=(5,),
                                           low = -1.0,
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

        self.marker2 = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                   rgba_color=[1, 1, 0, 1],
                                   radius=0.1,
                                   length=0.1,
                                   initial_offset=[0, 0, 0.1 / 2.0])
        self.marker2.load()

        self.mp_loaded = True

    def plan_base_motion(self, x,y,theta):
        half_size = self.map_size / 2.0
        path = plan_base_motion(self.robot_id, [x,y,theta], ((-half_size, -half_size), (half_size, half_size)), obstacles=[self.mesh_id])
        return path

    def step(self, pt):
        # point = [x,y]
        x = int((pt[0]+1)/2.0 * 150)
        y = int((pt[1]+1)/2.0 * 128)
        yaw = self.robots[0].get_rpy()[2]
        orn = pt[2] * np.pi + yaw

        armx = int((pt[3]+1)/2.0 * 128)
        army = int((pt[4]+1)/2.0 * 128)

        opos = get_base_values(self.robot_id)

        self.get_additional_states()
        org_potential = self.get_potential()

        state, _, done, _ = super(MotionPlanningBaseArmEnv, self).step(None)
        points = state['pc']

        if x < 128:

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
            set_base_values(self.robot_id, [bq[0], bq[1], bq[2]])
            state, _, done, info = super(MotionPlanningBaseArmEnv, self).step(None)
            self.get_additional_states()

        else:
            set_base_values(self.robot_id, opos)
            state, _, done, info = super(MotionPlanningBaseArmEnv, self).step(None)
            reward = -0.02

            del state['pc']
            return state, reward, done, info

        point = points[armx, army]
        camera_pose = (self.robots[0].parts['eyes'].get_pose())
        transform_mat = quat_pos_to_mat(pos=camera_pose[:3],
                                        quat=[camera_pose[6], camera_pose[3], camera_pose[4], camera_pose[5]])
        projected_point2 = (transform_mat).dot(np.array([-point[2], -point[0], point[1], 1]))[:3]

        projected_point2[2] += 1 # 1m from floor

        self.marker2.set_position(projected_point2)

        robot_id = self.robot_id
        arm_joints = joints_from_names(robot_id, ['torso_lift_joint', 'shoulder_pan_joint', 'shoulder_lift_joint',
                                                  'upperarm_roll_joint',
                                                  'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint',
                                                  'wrist_roll_joint'])
        max_limits = [0, 0] + get_max_limits(robot_id, arm_joints) + [0.05, 0.05]
        min_limits = [0, 0] + get_min_limits(robot_id, arm_joints) + [0, 0]
        rest_position = [0, 0] + list(get_joint_positions(robot_id, arm_joints)) + [0.04, 0.04]
        joint_range = list(np.array(max_limits) - np.array(min_limits))
        joint_range = [item + 1 for item in joint_range]
        jd = [0.1 for item in joint_range]
        #
        jp = p.calculateInverseKinematics(self.robot_id, self.robots[0].parts['gripper_link'].body_part_index, projected_point2,
                                     lowerLimits=min_limits,
                                     upperLimits=max_limits,
                                     jointRanges=joint_range,
                                     restPoses=rest_position,
                                     jointDamping=jd,
                                     solver=p.IK_DLS,
                                     maxNumIterations=100)[2:10]

        print(jp)
        set_joint_positions(self.robot_id, arm_joints, jp)
        p1 = self.robots[0].get_end_effector_position()
        dist = np.linalg.norm(np.array(p1) - np.array(projected_point2))

        print(dist)
        rp = (0.38548146667743244, 1.1522793897208579, 1.2576467971105596, -0.312703569911879,
         1.7404867100093226, -0.0962895617312548, -1.4418232619629425, -1.6780152866247762)

        reward = org_potential - self.get_potential()

        if l2_distance(self.target_pos, self.robots[0].get_end_effector_position()) < self.dist_tol:
            reward += self.success_reward  # |success_reward| = 10.0 per step
            done = True
        else:
            done = False

        set_joint_positions(self.robot_id, arm_joints, rp)
        state, _, _, info = super(MotionPlanningBaseArmEnv, self).step(None)

        print('reward', reward)
        self.planner_step += 1

        if self.planner_step > self.max_step:
            done = True

        info['planner_step'] = self.planner_step
        del state['pc']

        return state, reward, done, info

    def reset(self):
        state = super(MotionPlanningBaseArmEnv, self).reset()

        if not self.mp_loaded:
            self.prepare_motion_planner()

        self.target_pos[2] = 1.0
        self.target_pos_vis_obj.set_position(self.target_pos)

        self.planner_step = 0
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
        nav_env.reset()
        for i in range(150):
            action = nav_env.action_space.sample()
            state, reward, done, info = nav_env.step(action)
            #print(state, reward, done, info)
            #time.sleep(0.05)
            #nav_env.step()
        #for step in range(50):  # 500 steps, 50s world time
        #    action = nav_env.action_space.sample()
        #    state, reward, done, _ = nav_env.step(action)
        #    # print('reward', reward)
            if done:
                print('Episode finished after {} timesteps'.format(i + 1))
                break
        print(time.time() - start)
    nav_env.clean()
