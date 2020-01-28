from gibson2.core.physics.interactive_objects import VisualObject, InteractiveObj, VisualMarker, BoxShape, CylinderPedestrian, Pedestrian
import gibson2
from gibson2.utils.utils import rotate_vector_3d, l2_distance, quatToXYZW, parse_config
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
import random


from gibson2.core.pedestrians.human import Human
from gibson2.core.pedestrians.robot import Robot
from gibson2.core.pedestrians.state import ObservableState

import collections


Episode = collections.namedtuple('Episode',
                                 ['env',
                                  'agent',
                                  'initial_pos',
                                  'target_pos',
                                  'geodesic_distance',
                                  'shortest_path',
                                  'agent_trajectory',
                                  'object_files',
                                  'object_trajectory',
                                  'success',
                                  'path_efficiency',
                                  'kinematic_disturbance',
                                  'dynamic_disturbance_a',
                                  'dynamic_disturbance_b',
                                  'collision_step',
                                  ])

class NavigateEnv(BaseEnv):
    """
    We define navigation environments following Anderson, Peter, et al. 'On evaluation of embodied navigation agents.'
    arXiv preprint arXiv:1807.06757 (2018). (https://arxiv.org/pdf/1807.06757.pdf)

    """
    def __init__(
            self,
            config_file,
            model_id=None,
            mode='headless',
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            automatic_reset=False,
            device_idx=0,
    ):
        super(NavigateEnv, self).__init__(config_file=config_file,
                                          model_id=model_id,
                                          mode=mode,
                                          device_idx=device_idx)
        self.automatic_reset = automatic_reset

        # simulation
        self.mode = mode
        self.action_timestep = action_timestep
        self.physics_timestep = physics_timestep
        self.simulator.set_timestep(physics_timestep)
        self.simulator_loop = int(self.action_timestep / self.simulator.timestep)
        self.current_step = 0
        self.path_length = 0.0
        self.agent_trajectory = []
        self.stage = None
        self.current_episode = 0
        self.floor_num = None
        self.num_object_classes = None
        self.stored_episodes = collections.deque(maxlen=100)
        self.interactive_objects = []
        # self.reward_stats = []
        # self.state_stats = {'sensor': [], 'auxiliary_sensor': []}
        
        self.n_steps = 0
        self.n_successes = 0
        self.n_collisions = 0
        self.n_ped_collisions = 0
        self.n_ped_hits_robot = 0
        self.n_timeouts = 0
        self.n_personal_space_violations = 0
        self.n_cutting_off = 0
        self.ped_collision = False
        self.success = False
        self.distance_traveled = 0.0
        self.time_elapsed = 0.0
        self.episode_distance = 0.0
        self.spl_sum = 0 # shortest path length (SPL)
        self.spl = 0     # average shortest path length       

    def load(self):
        super(NavigateEnv, self).load()
        self.initial_pos = np.array(self.config.get('initial_pos', [0, 0, 0]))
        self.initial_orn = np.array(self.config.get('initial_orn', [0, 0, 0]))

        self.target_pos = np.array(self.config.get('target_pos', [5, 5, 0]))
        self.target_orn = np.array(self.config.get('target_orn', [0, 0, 0]))

        self.additional_states_dim = self.config.get('additional_states_dim', 0)
        self.auxiliary_sensor_dim = self.config.get('auxiliary_sensor_dim', 0)
        self.normalize_observation = self.config.get('normalize_observation', False)
        self.observation_normalizer = self.config.get('observation_normalizer', {})
        for key in self.observation_normalizer:
            self.observation_normalizer[key] = np.array(self.observation_normalizer[key])

        # termination condition
        self.stage = 0
        self.dist_tol = self.config.get('dist_tol', 0.2)
        self.max_step = self.config.get('max_step', float('inf'))

        # reward
        self.reward_type = self.config.get('reward_type', 'dense')
        assert self.reward_type in ['dense', 'sparse', 'l2', 'stage_sparse']

        self.success_reward = self.config.get('success_reward', 10.0)
        self.slack_reward = self.config.get('slack_reward', -0.01)
        self.death_z_thresh = self.config.get('death_z_thresh', 0.1)

        # reward weight
        self.potential_reward_weight = self.config.get('potential_reward_weight', 10.0)
        self.electricity_reward_weight = self.config.get('electricity_reward_weight', 0.0)
        self.stall_torque_reward_weight = self.config.get('stall_torque_reward_weight', 0.0)
        self.collision_reward_weight = self.config.get('collision_reward_weight', 0.0)
        # ignore the agent's collision with these body ids, typically ids of the ground and the robot itself
        # self.collision_ignore_body_ids = self.config.get('collision_ignore_body_ids', [])

# Master =======
#         # ignore the agent's collision with these body ids, typically ids of the ground
        self.collision_ignore_body_b_ids = set(self.config.get('collision_ignore_body_b_ids', self.scene_ids))
        self.collision_ignore_link_a_ids = set(self.config.get('collision_ignore_link_a_ids', []))

        # discount factor
        self.discount_factor = self.config.get('discount_factor', 1.0)
        self.output = self.config['output']
        self.n_horizontal_rays = self.config.get('n_horizontal_rays', 128)
        self.n_vertical_beams = self.config.get('n_vertical_beams', 9)
        

        # obstacles
        self.num_obstacles = self.config.get('num_obstacles', 0)        

        # pedestrians
        self.num_pedestrians = self.config.get('num_pedestrians', 0)
        self.pedestrians_can_see_robot = self.config['pedestrians_can_see_robot']        
        self.randomize_pedestrian_attributes = self.config.get('randomize_pedestrian_attributes', False)

        # TODO: sensor: observations that are passed as network input, e.g. target position in local frame
        # TODO: auxiliary sensor: observations that are not passed as network input, but used to maintain the same
        # subgoals for the next T time steps, e.g. agent pose in global frame
        self.sensor_dim = self.additional_states_dim
        self.action_dim = self.robots[0].action_dim
        
        observation_space = OrderedDict()
        if 'concatenate' in self.output:
            self.concatenate_space = gym.spaces.Box(low=-1.0,
                                           high=1.0,
                                           shape=(2 + self.config['n_horizontal_rays'],),
                                           dtype=np.float32)
            
            observation_space['concatenate'] = self.concatenate_space            
        
        if 'sensor' in self.output:
            self.sensor_space = gym.spaces.Box(low=-np.inf,
                                               high=np.inf,
                                               shape=(self.sensor_dim,),
                                               dtype=np.float32)
            observation_space['sensor'] = self.sensor_space
        if 'auxiliary_sensor' in self.output:
            self.auxiliary_sensor_space = gym.spaces.Box(low=-np.inf,
                                                         high=np.inf,
                                                         shape=(self.auxiliary_sensor_dim,),
                                                         dtype=np.float32)
            observation_space['auxiliary_sensor'] = self.auxiliary_sensor_space
        if 'pointgoal' in self.output:
            self.pointgoal_space = gym.spaces.Box(low=-np.inf,
                                                  high=np.inf,
                                                  shape=(2,),
                                                  dtype=np.float32)
            observation_space['pointgoal'] = self.pointgoal_space
        if 'rgb' in self.output:
            self.rgb_space = gym.spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.config.get('resolution', 64),
                                                   self.config.get('resolution', 64),
                                                   3),
                                            dtype=np.float32)
            observation_space['rgb'] = self.rgb_space
        if 'depth' in self.output:
            self.depth_space = gym.spaces.Box(low=-np.inf,
                                              high=np.inf,
                                              shape=(self.config.get('resolution', 64),
                                                     self.config.get('resolution', 64),
                                                     1),
                                              dtype=np.float32)
            observation_space['depth'] = self.depth_space
        # if 'scan' in self.output:
        #     self.scan_space = gym.spaces.Box(low=-np.inf,
        #                                       high=np.inf,
        #                                      shape=(self.config.get('resolution', 64),
        #                                              1),
        #                                       dtype=np.float32)
        #     observation_space['scan'] = self.scan_space
        if 'scan' in self.output:
            self.scan_space = gym.spaces.Box(low=-np.inf,
                                              high=np.inf,
                                              shape=(self.config.get('resolution', 64),
                                                     1),
                                              dtype=np.float32)
            observation_space['scan'] = self.scan_space
            
        if 'seg' in self.output:
            self.seg_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.config.get('resolution', 64),
                                                   self.config.get('resolution', 64),
                                                   1),
                                            dtype=np.float32)
            observation_space['seg'] = self.seg_space
        if 'depth_seg' in self.output:
            self.depth_seg_space = gym.spaces.Box(low=0.0,
                                                  high=1.0,
                                                  shape=(self.config.get('resolution', 64),
                                                         self.config.get('resolution', 64),
                                                         2),
                                                  dtype=np.float32)
            observation_space['depth_seg'] = self.depth_seg_space
        if 'scan' in self.output:
            self.scan_space = gym.spaces.Box(low=-np.inf,
                                             high=np.inf,
                                             shape=(self.n_horizontal_rays * self.n_vertical_beams, 3),
                                             dtype=np.float32)
            observation_space['scan'] = self.scan_space
            
        if 'rgb_filled' in self.output:  # use filler
            self.comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
            self.comp = torch.nn.DataParallel(self.comp).cuda()
            self.comp.load_state_dict(
                torch.load(os.path.join(gibson2.assets_path, 'networks', 'model.pth')))
            self.comp.eval()

        if 'pedestrian' in self.output:
            self.pedestrian_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                   shape=(self.num_pedestrians*2,),  # num_pedestrians * len([x_pos, y_pos])
                                                   dtype=np.float32)
            observation_space['pedestrian'] = self.pedestrian_space
            
        if 'pedestrian_position' in self.output:
            # print('NUMBER OF PEDESTRIANS called once: {}'.format(self.num_pedestrians))
            self.pedestrian_position_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                   shape=(self.num_pedestrians*2,),  # num_pedestrians * len([x_pos, y_pos])
                                                   dtype=np.float32)
            observation_space['pedestrian_position'] = self.pedestrian_position_space
            
        if 'pedestrian_velocity' in self.output:
            self.pedestrian_velocity_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                   shape=(self.num_pedestrians*2,),  # num_pedestrians * len([x_pos, y_pos])
                                                   dtype=np.float32)
            observation_space['pedestrian_velocity'] = self.pedestrian_velocity_space
            
        if 'waypoints' in self.output:
            self.waypoints_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                  shape=(self.config['waypoints']*2,),  # waypoints * len([x_pos, y_pos])
                                                  dtype=np.float32)
            observation_space['waypoints'] = self.waypoints_space

        self.observation_space = gym.spaces.Dict(observation_space)
        self.action_space = self.robots[0].action_space

        # add visual objects

        self.visual_object_at_initial_pos = self.config.get(
            'visual_object_at_initial_pos', False)

        self.visual_object_at_target_pos = self.config.get(
            'visual_object_at_target_pos', False)

        if self.visual_object_at_initial_pos:
            cyl_length = 1.2
            self.initial_pos_vis_obj = VisualObject(visual_shape=p.GEOM_CYLINDER,
                                                    rgba_color=[0, 0, 1, 0.4],
                                                    radius=0.3,
                                                    length=cyl_length,
                                                    initial_offset=[0, 0, cyl_length / 2.0])
            self.initial_pos_vis_obj.load()            

        if self.visual_object_at_target_pos:
            cyl_length = 1.2            
            self.target_pos_vis_obj = VisualObject(visual_shape=p.GEOM_CYLINDER,
                                                   rgba_color=[0, 1, 0, 0.6],
                                                   radius=0.3,
                                                   length=cyl_length,
                                                   initial_offset=[0, 0, cyl_length / 2.0])

        self.visual_object_at_initial_target_pos = self.config.get('visual_object_at_initial_target_pos', False)
#        if self.visual_object_at_initial_target_pos:
#            cyl_length = 0.2
#            self.initial_pos_vis_obj = VisualMarker(visual_shape=p.GEOM_CYLINDER,
#                                                    rgba_color=[1, 0, 0, 0.3],
#                                                    radius=self.dist_tol,
#                                                    length=cyl_length,
#                                                    initial_offset=[0, 0, cyl_length / 2.0])
#            self.target_pos_vis_obj = VisualMarker(visual_shape=p.GEOM_CYLINDER,
#                                                   rgba_color=[0, 1, 0, 0.3],
#                                                   radius=self.dist_tol,
#                                                  length=cyl_length,
#                                                   initial_offset=[0, 0, cyl_length / 2.0])



        if self.config.get('target_visual_object_visible_to_agent', False):
            self.simulator.import_object(self.target_pos_vis_obj, class_id=255)
        else:
            self.target_pos_vis_obj.load()
    
    def get_additional_states(self):
        relative_position = self.current_target_position - self.robots[0].get_position()
        # rotate relative position back to body point of view
        additional_states = rotate_vector_3d(relative_position, *self.robots[0].get_rpy())
        additional_states = additional_states[0:2]
        
        if self.config['task'] == 'reaching':
            end_effector_pos = self.robots[0].get_end_effector_position() - self.robots[0].get_position()
            end_effector_pos = rotate_vector_3d(end_effector_pos, *self.robots[0].get_rpy())
            additional_states = np.concatenate((additional_states, end_effector_pos))
        assert len(additional_states) == self.additional_states_dim, 'additional states dimension mismatch'

        return additional_states

    def get_auxiliary_sensor(self, collision_links=[]):
        return np.array([])

    def get_state(self, collision_links=[]):
        # calculate state
        sensor_state = self.get_additional_states()
        auxiliary_sensor = self.get_auxiliary_sensor(collision_links)

        # rgb = self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
        # depth = np.clip(depth / 5.0, 0.0, 1.0)
        # depth = 1.0 - depth  # flip black/white
        # seg = self.simulator.renderer.render_robot_cameras(modes='seg')[0][:, :, 0:1]
        # if self.num_object_classes is not None:
        #     seg = np.clip(seg * 255.0 / self.num_object_classes, 0.0, 1.0)
        # cv2.imshow('rgb', rgb)
        # cv2.imshow('depth', depth)
        # cv2.imshow('seg', seg)

        state = OrderedDict()
        if 'sensor' in self.output:
            state['sensor'] = sensor_state
        if 'auxiliary_sensor' in self.output:
            state['auxiliary_sensor'] = auxiliary_sensor
        if 'pointgoal' in self.output:
            state['pointgoal'] = sensor_state[:2]
        if 'rgb' in self.output:
            state['rgb'] = self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]
        if 'depth' in self.output:
            depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
            state['depth'] = depth
        if 'normal' in self.output:
            state['normal'] = self.simulator.renderer.render_robot_cameras(modes='normal')
        if 'seg' in self.output:
            seg = self.simulator.renderer.render_robot_cameras(modes='seg')[0][:, :, 0:1]
            if self.num_object_classes is not None:
                seg = np.clip(seg * 255.0 / self.num_object_classes, 0.0, 1.0)
            state['seg'] = seg
        if 'depth_seg' in self.output:
            depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
            depth = np.clip(depth / 5.0, 0.0, 1.0)
            seg = self.simulator.renderer.render_robot_cameras(modes='seg')[0][:, :, 0:1]
            if self.num_object_classes is not None:
                seg = np.clip(seg * 255.0 / self.num_object_classes, 0.0, 1.0)
            depth_seg = np.concatenate((depth, seg), axis=2)
            state['depth_seg'] = depth_seg
        if 'rgb_filled' in self.output:
            with torch.no_grad():
                tensor = transforms.ToTensor()((state['rgb'] * 255).astype(np.uint8)).cuda()
                rgb_filled = self.comp(tensor[None, :, :, :])[0].permute(1, 2, 0).cpu().numpy()
                state['rgb_filled'] = rgb_filled

        if 'pointgoal' in self.output:
            state['pointgoal'] = sensor_state[:2]

        # if 'scan' in self.output:
            # depth_lidar = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
            # state['scan'] = np.amin(depth_lidar, axis=1)

        # TODO: figure out why 'scan' consumes so much cpu
        if 'scan' in self.output:
            """
            assert 'scan_link' in self.robots[0].parts, "Requested scan but no scan_link"
            pose_camera = self.robots[0].parts['scan_link'].get_pose()
            angle = np.arange(0, 2 * np.pi, 2 * np.pi / float(self.n_horizontal_rays))
            elev_bottom_angle = -30. * np.pi / 180.
            elev_top_angle = 10. * np.pi / 180.
            elev_angle = np.arange(elev_bottom_angle, elev_top_angle,
                                   (elev_top_angle - elev_bottom_angle) / float(self.n_vertical_beams))
            orig_offset = np.vstack([
                np.vstack([np.cos(angle),
                           np.sin(angle),
                           np.repeat(np.tan(elev_ang), angle.shape)]).T for elev_ang in elev_angle
            ])
            transform_matrix = quat2mat([pose_camera[-1], pose_camera[3], pose_camera[4], pose_camera[5]])
            offset = orig_offset.dot(np.linalg.inv(transform_matrix))
            pose_camera = pose_camera[None, :3].repeat(self.n_horizontal_rays * self.n_vertical_beams, axis=0)

            results = p.rayTestBatch(pose_camera, pose_camera + offset * 30)
            hit = np.array([item[0] for item in results])
            dist = np.array([item[2] for item in results])

            valid_pts = (dist < 1. - 1e-5) & (dist > 0.1 / 30) & (hit != self.robots[0].robot_ids[0]) & (hit != -1)
            dist[~valid_pts] = 1.0  # zero out invalid pts
            dist *= 30

            xyz = np.expand_dims(dist, 1) * orig_offset
            """
            assert 'scan_link' in self.robots[0].parts, "Requested scan but no scan_link"
            pose_camera = self.robots[0].parts['scan_link'].get_pose()
            angle = np.arange(0, 2 * np.pi, 2 * np.pi / float(self.n_horizontal_rays))
            elev_bottom_angle = -15. * np.pi / 180.
            elev_top_angle = 15. * np.pi / 180.
            if self.n_vertical_beams > 1:
                elev_angle = np.arange(elev_bottom_angle, elev_top_angle,
                                   (elev_top_angle - elev_bottom_angle) / float(self.n_vertical_beams))
            
            else:
                elev_angle = [0.0]
            orig_offset = np.vstack([
                np.vstack([np.cos(angle),
                           np.sin(angle),
                           np.repeat(np.tan(elev_ang), angle.shape)]).T for elev_ang in elev_angle
            ])
            transform_matrix = quat2mat([pose_camera[-1], pose_camera[3], pose_camera[4], pose_camera[5]])
            offset = orig_offset.dot(np.linalg.inv(transform_matrix))
            pose_camera = pose_camera[None, :3].repeat(self.n_horizontal_rays * self.n_vertical_beams, axis=0)

            results = p.rayTestBatch(pose_camera, pose_camera + offset * 30)
            hit = np.array([item[0] for item in results])
            dist = np.array([item[2] for item in results])
                        
            valid_pts = (dist < 1. - 1e-5) & (dist > 0.1 / 30) & (hit != self.robots[0].robot_ids[0]) & (hit != -1)
            dist[~valid_pts] = 1.0  # set invalid points to max range
            
            dist *= 30
            
            xyz = np.expand_dims(dist, 1) * orig_offset
            #state_scan = xyz
            
            # state_scan = dist
            state['scan'] = xyz


            assert laser_link_name in self.robots[0].parts, "Requested scan but no scan_link"

            laser_angular_half_range = laser_angular_range / 2.0
            laser_pose = self.robots[0].parts[laser_link_name].get_pose()

            # self.scan_vis.set_position(pos=laser_pose[:3])

            transform_matrix = quat2mat([laser_pose[6], laser_pose[3], laser_pose[4], laser_pose[5]])  # [x, y, z, w]
            angle = np.arange(-laser_angular_half_range / 180 * np.pi,
                              laser_angular_half_range / 180 * np.pi,
                              laser_angular_range / 180.0 * np.pi / self.n_horizontal_rays)
            unit_vector_local = np.array([[np.cos(ang), np.sin(ang), 0.0] for ang in angle])
            unit_vector_world = transform_matrix.dot(unit_vector_local.T).T
            start_pose = np.tile(laser_pose[:3], (self.n_horizontal_rays, 1))
            start_pose += unit_vector_world * min_laser_dist
            end_pose = laser_pose[:3] + unit_vector_world * laser_linear_range
            results = p.rayTestBatch(start_pose, end_pose, 6)  # numThreads = 6
            # hit_object_id = np.array([item[0] for item in results])
            # link_id = np.array([item[1] for item in results])
            hit_fraction = np.array([item[2] for item in results])  # hit fraction = [0.0, 1.0] of laser_linear_range
            state['scan'] = np.expand_dims(hit_fraction, 1)
            state['scan'] *= laser_linear_range

        if 'pedestrian' in self.output:
            ped_pos = self.get_ped_states()
            rob_pos = self.robots[0].get_position()
            ped_robot_relative_pos = [[ped_pos[i][0] - rob_pos[0], ped_pos[i][1] - rob_pos[1]] for i in range(self.num_pedestrians)]
            ped_robot_relative_pos = np.asarray(ped_robot_relative_pos).flatten()
            state['pedestrian'] = ped_robot_relative_pos # [x1, y1, x2, y2,...] in robot frame
            
        if 'pedestrian_position' in self.output:
            ped_pos = self.get_ped_positions()
            rob_pos = self.robots[0].get_position()
            ped_robot_relative_pos = [rotate_vector_3d([ped_pos[i][0] - rob_pos[0], ped_pos[i][1] - rob_pos[1], 0], *self.robots[0].get_rpy())[0:2] for i in range(self.num_pedestrians)]
            # Crowdsim normalization for direct comparison
            ped_robot_relative_pos = np.asarray(ped_robot_relative_pos).flatten()
            state['pedestrian_position'] = ped_robot_relative_pos # [x1, y1, x2, y2,...] in robot frame
            
        if 'pedestrian_velocity' in self.output:
            ped_vel = self.get_ped_velocities()
            rob_vel = self.robots[0].get_velocity()
            ped_robot_relative_vel = [rotate_vector_3d([ped_vel[i][0] - rob_vel[0], ped_vel[i][1] - rob_vel[1], 0], *self.robots[0].get_rpy())[0:2] for i in range(self.num_pedestrians)]
            # Crowdsim normalization for direct comparison
            ped_robot_relative_vel = np.asarray(ped_robot_relative_vel).flatten()
            state['pedestrian_velocity'] = ped_robot_relative_vel # [vx1, vy1, vx2, vy2,...] in robot frame
        
        if 'pedestrian_ttc' in self.output:
            ped_ttc = self.get_ped_time_to_collision()
            ped_robot_relative_ttc = np.asarray(ped_ttc).flatten()
            state['pedestrian_ttc'] = ped_robot_relative_ttc # [ttc1, ttc2, ...] in robot frame
            
        if 'waypoints' in self.output:
            path = self.compute_a_star(self.config['scene']) # current dim is (107, 2), varying by scene and start/end points
            rob_pos = self.robots[0].get_position()
            path_robot_relative_pos = [[path[i][0] - rob_pos[0], path[i][1] - rob_pos[1]] for i in range(path.shape[0])]
            path_robot_relative_pos = np.asarray(path_robot_relative_pos)
            path_point_ind = np.argmin(np.linalg.norm(path_robot_relative_pos , axis=1))
            curr_points_num = path.shape[0] - path_point_ind
            # keep the dimenstion based on the number of waypoints specified in the config file
            if curr_points_num > self.config['waypoints']:
                out = path_robot_relative_pos[path_point_ind:path_point_ind+self.config['waypoints']]
            else:
                curr_waypoints = path_robot_relative_pos[path_point_ind:]
                end_point = np.repeat(path_robot_relative_pos[path.shape[0]-1].reshape(1,2), (self.config['waypoints']-curr_points_num), axis=0)
                out = np.vstack((curr_waypoints, end_point))
            state['waypoints'] = out.flatten()

        if 'concatenate' in self.output:
            # TODO: figure out why 'scan' consumes so much cpu
            assert 'scan_link' in self.robots[0].parts, "Requested scan but no scan_link"
            pose_camera = self.robots[0].parts['scan_link'].get_pose()
            angle = np.arange(0, 2 * np.pi, 2 * np.pi / float(self.n_horizontal_rays))
            elev_bottom_angle = -15. * np.pi / 180.
            elev_top_angle = 15. * np.pi / 180.
            if self.n_vertical_beams > 1:
                elev_angle = np.arange(elev_bottom_angle, elev_top_angle,
                                   (elev_top_angle - elev_bottom_angle) / float(self.n_vertical_beams))
            
            else:
                elev_angle = [0.0]
            orig_offset = np.vstack([
                np.vstack([np.cos(angle),
                           np.sin(angle),
                           np.repeat(np.tan(elev_ang), angle.shape)]).T for elev_ang in elev_angle
            ])
            transform_matrix = quat2mat([pose_camera[-1], pose_camera[3], pose_camera[4], pose_camera[5]])
            offset = orig_offset.dot(np.linalg.inv(transform_matrix))
            pose_camera = pose_camera[None, :3].repeat(self.n_horizontal_rays * self.n_vertical_beams, axis=0)

            results = p.rayTestBatch(pose_camera, pose_camera + offset * 30)
            hit = np.array([item[0] for item in results])
            dist = np.array([item[2] for item in results])
                        
            valid_pts = (dist < 1. - 1e-5) & (dist > 0.1 / 30) & (hit != self.robots[0].robot_ids[0]) & (hit != -1)
            dist[~valid_pts] = 1.0  # set invalid points to max range
            
            dist *= 30
            
            #xyz = np.expand_dims(dist, 1) * orig_offset
            #state_scan = xyz
            
            state_scan = dist

            state['concatenate'] = np.concatenate([sensor_state[0:2], state_scan], axis=None).flatten()

            state['concatenate'] /= 30.0 # normalize by the max lidar range
        return state

        # if 'concatenate' in self.output:
        #     normalizer = 12.0 / np.sqrt(2.0)
            
        #     sensor_state /= normalizer
            
        #     ped_pos = self.get_ped_positions()
        #     rob_pos = self.robots[0].get_position()
        #     ped_robot_relative_pos = [rotate_vector_3d([ped_pos[i][0] - rob_pos[0], ped_pos[i][1] - rob_pos[1], 0], *self.robots[0].get_rpy())[0:2] for i in range(self.num_pedestrians)]
        #     ped_robot_relative_pos /= normalizer
        #     ped_robot_relative_pos = np.asarray(ped_robot_relative_pos).flatten()
        #     state_pedestrian_position = ped_robot_relative_pos # [x1, y1, x2, y2,...] in robot frame

        #     ped_vel = self.get_ped_velocities()
        #     rob_vel = self.robots[0].get_velocity()
        #     ped_robot_relative_vel = [rotate_vector_3d([ped_vel[i][0] - rob_vel[0], ped_vel[i][1] - rob_vel[1], 0], *self.robots[0].get_rpy())[0:2] for i in range(self.num_pedestrians)]
        #     ped_robot_relative_vel /= normalizer
        #     ped_robot_relative_vel = np.asarray(ped_robot_relative_vel).flatten()
        #     state_pedestrian_velocity = ped_robot_relative_vel # [vx1, vy1, vx2, vy2,...] in robot frame

        #     ped_ttc = self.get_ped_time_to_collision()
        #     ped_robot_relative_ttc = np.asarray(ped_ttc).flatten()
        #     state_pedestrian_ttc = ped_robot_relative_ttc # [ttc1, ttc2, ...] in robot frame
            
        #     state['concatenate'] = np.concatenate([sensor_state[0:2], state_pedestrian_position, state_pedestrian_velocity, state_pedestrian_ttc], axis=None).flatten()
        # return state

    def get_ped_states(self):
        return [(self.humans[i].px, self.humans[i].py) for i in range(self.num_pedestrians)]
    
    def get_ped_positions(self):
        return [(self.humans[i].px, self.humans[i].py) for i in range(self.num_pedestrians)]
    
    def get_ped_velocities(self):
        return [(self.humans[i].vx, self.humans[i].vy) for i in range(self.num_pedestrians)]
    
    def get_ped_time_to_collision(self):
        ped_pos = self.get_ped_positions()
        rob_pos = self.robots[0].get_position()
        ped_robot_relative_pos = [rotate_vector_3d([ped_pos[i][0] - rob_pos[0], ped_pos[i][1] - rob_pos[1], 0], *self.robots[0].get_rpy())[0:2] for i in range(self.num_pedestrians)]

        ped_vel = self.get_ped_velocities()
        rob_vel = self.robots[0].get_velocity()
        ped_robot_relative_vel = [rotate_vector_3d([ped_vel[i][0] - rob_vel[0], ped_vel[i][1] - rob_vel[1], 0], *self.robots[0].get_rpy())[0:2] for i in range(self.num_pedestrians)]
        
        ttc = list()
        
        for pos, vel in zip(ped_pos, ped_vel):
            if (vel[0] * pos[0] + vel[1] * pos[1]) == 0:
                time_to_collision = -1.0
            else:
                time_to_collision = -1.0 * (pos[0]**2 + pos[1]**2) / (vel[0] * pos[0] + vel[1] * pos[1])
        
            if time_to_collision < 0:
                time_to_collision = -1.0
            else:
                time_to_collision = 1.0 - np.tanh(time_to_collision / 10.0)
        
            ttc.append(time_to_collision)
                
        return ttc

    def run_simulation(self):
        collision_links = []
        for _ in range(self.simulator_loop):
            self.simulator_step()

            if self.num_pedestrians > 0:
                self.update_pedestrian_positions_in_gibson()
                
            collision_links.append(list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0])))

        return self.filter_collision_links(collision_links)

#     def prevent_stuck_at_corners(self, x, y, old_x, old_y, eps = 0.01):
#         # self.pedestrian_simulator.setAgentPosition(ai, (self._ped_states[ai,0], self._ped_states[ai,1]))
#         def dist(x1, y1, x2, y2, eps = 0.01):
#             return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
# 
#         for i in range(self.num_pedestrians):
#             if(dist(x[i], y[i], old_x[i], old_y[i]) < eps):
#                 pedestrian = self.pedestrian_ids[i]
#                 assig_speed = np.random.uniform(0.001, 0.01)
#                 assig_direc = np.random.uniform(0.0, 2*np.pi)
#                 vx = assig_speed * np.cos(assig_direc)
#                 vy = assig_speed * np.sin(assig_direc)
#                 self.pedestrian_simulator.setAgentPrefVelocity(pedestrian, (vx, vy))
#                 self.pedestrian_simulator.setAgentVelocity(pedestrian, (vx, vy))

    # def filter_collision_links(self, collision_links):
    # return [elem for elem in collision_links if elem[0] not in self.collision_ignore_body_ids]
    def filter_collision_links(self, collision_links):
        # print('COLLISION LINKS: {}'.format(collision_links))
        #print('IGNORE LINK: {}'.format(self.collision_ignore_link_a_ids))
        filtered_collision_links = [[item for item in collision_per_sim_step
             if item[2] not in self.collision_ignore_body_b_ids and
             item[3] not in self.collision_ignore_link_a_ids]
            for collision_per_sim_step in collision_links]
        # print('FILTERED COLLISOIN LINKS: {}'.format(filtered_collision_links))
        return filtered_collision_links

    # def filter_collision_links(self, collision_links):
        # return [[item for item in collision_per_sim_step 
               #  if item[2] not in self.collision_ignore_body_b_ids and \
               # item[3] not in self.collision_ignore_link_a_ids]
               #  for collision_per_sim_step in collision_links]

    def get_position_of_interest(self):
        if self.config['task'] == 'pointgoal':
            return self.robots[0].get_position()
        elif self.config['task'] == 'reaching':
            return self.robots[0].get_end_effector_position()

    def get_potential(self):
        return l2_distance(self.current_target_position, self.get_position_of_interest())


    def get_l2_potential(self):
        return l2_distance(self.target_pos, self.get_position_of_interest())

    def get_reward(self, collision_links=[], action=None, info={}):
        collision_links_flatten = [item for sublist in collision_links for item in sublist]

        reward = self.slack_reward  # |slack_reward| = 0.01 per step

        if self.reward_type == 'l2':
            new_potential = self.get_l2_potential()
        elif self.reward_type == 'dense':
            new_potential = self.get_potential()
        potential_reward = self.potential - new_potential
        reward += potential_reward * self.potential_reward_weight  # |potential_reward| ~= 0.1 per step
        self.potential = new_potential

        # electricity_reward = np.abs(self.robots[0].joint_speeds * self.robots[0].joint_torque).mean().item()
        electricity_reward = 0.0
        reward += electricity_reward * self.electricity_reward_weight  # |electricity_reward| ~= 0.05 per step

        # stall_torque_reward = np.square(self.robots[0].joint_torque).mean()
        stall_torque_reward = 0.0
        reward += stall_torque_reward * self.stall_torque_reward_weight  # |stall_torque_reward| ~= 0.05 per step

        collision_reward = float(len(collision_links) > 0)
        if collision_reward > 0:
            self.collision = True
            info['collision_reward'] = collision_reward * self.collision_reward_weight  # expose collision reward to info
            reward += collision_reward * self.collision_reward_weight  # |collision_reward| ~= 1.0 per step if collision
        else:
            self.collision = False

        collision_reward = float(len(collision_links_flatten) > 0)
        self.collision_step += int(len(collision_links_flatten) > 0)
        info['collision_reward'] = collision_reward * self.collision_reward_weight  # expose collision reward to info
        reward += collision_reward * self.collision_reward_weight  # |collision_reward| ~= 1.0 per step if collision

        # goal reached
        if l2_distance(self.current_target_position, self.get_position_of_interest()) < self.dist_tol:
            reward += self.success_reward  # |success_reward| = 10.0 per step

        return reward, info

    def get_termination(self, collision_links=[], info={}):
        done = False
        # for item in collision_links_flatten:
        #     if item[9] > 500:
        #         print("collision between " + self.id_to_name[self.robots[0].robot_ids[0]]["links"][item[3]]
        #               + " and " + self.id_to_name[item[2]]["links"][item[4]])

        # door_angle = p.getJointState(self.door.body_id, self.door_axis_link_id)[0]
        # max_force = max([item[9] for item in collision_links_flatten]) if len(collision_links_flatten) > 0 else 0

        floor_height = 0.0 if self.floor_num is None else self.scene.get_floor_height(self.floor_num)
        
        if self.collision:
            print("COLLISION!")
            assert self.current_step > 0, "Collision shouldn't be penalized during resetting!"
            done = True
            info['success'] = False
            
            # don't blame the robot if it wasn't moving very fast
            robot_velocity = self.robots[0].get_velocity()

            if np.linalg.norm([robot_velocity[0], robot_velocity[1]]) < 0.05:
                self.n_ped_hits_robot += 1
            else:
                self.n_ped_collisions += 1
                self.n_collisions += 1
            
        elif l2_distance(self.current_target_position, self.get_position_of_interest()) < self.dist_tol:
            print("SUCCESS!")
        # if l2_distance(self.target_pos, self.get_position_of_interest()) < self.dist_tol:
            # print("GOAL")

            done = True
            info['success'] = True
            self.n_successes += 1

        # What is this branch?
        elif self.robots[0].get_position()[2] > floor_height + self.death_z_thresh:
            print("DEATH")
            done = True
            info['success'] = False

        elif self.current_step >= self.max_step:
            done = True
            print("TIMEOUT!")
            info['success'] = False
            self.n_timeouts += 1

        if done:
            info['episode_length'] = self.current_step
            info['path_length'] = self.path_length
            info['collision_step'] = self.collision_step
            info['energy_cost'] = self.energy_cost
            info['stage'] = self.stage

            info['episodes'] = self.current_episode
            info['successes'] = self.n_successes,
            info['collisions'] = self.n_collisions,
            info['ped_collisions'] = self.n_ped_collisions,
            info['ped_hits_robot'] = self.n_ped_hits_robot,
            info['timeouts'] = self.n_timeouts,
            info['personal_space_violations'] = 0 if self.distance_traveled == 0 else self.n_personal_space_violations / self.distance_traveled,
            info['cutting_off'] = 0 if self.distance_traveled == 0 else self.n_cutting_off / self.distance_traveled,
            info['success_rate'] = 0 if self.current_episode == 0 else 100 * self.n_successes / self.current_episode,
            info['collision_rate'] = 0 if self.current_episode == 0 else 100 * self.n_collisions / self.current_episode,
            info['ped_collision_rate'] = 0 if self.current_episode == 0 else 100 * self.n_ped_collisions / self.current_episode,
            info['ped_hits_robot_rate'] = 0 if self.current_episode == 0 else 100 * self.n_ped_hits_robot / self.current_episode,
            info['timeout_rate'] = 0 if self.current_episode == 0 else 100 * self.n_timeouts / self.current_episode,
            info['shortest_path_length'] = None if self.current_episode == 0 else [self.spl]

            # Episode = collections.namedtuple('Episode',
            #                                  ['initial_pos',
            #                                   'target_pos',
            #                                   'geodesic_distance',
            #                                   'shortest_path',
            #                                   'trajectory',
            #                                   'success',
            #                                   'path_efficiency',
            #                                   'kinematic_disturbance',
            #                                   'dynamic_disturbance_a',
            #                                   'dynamic_disturbance_b',
            #                                   'collision_step',
            #                                   ])

            # shortest_path, geodesic_distance = self.scene.get_shortest_path(self.floor_num,
            #                                                                 self.initial_pos[:2],
            #                                                                 self.target_pos[:2],
            #                                                                 entire_path=True)
            # floor_height = self.scene.get_floor_height(self.floor_num)
            # shortest_path = np.array([np.array([path[0], path[1], floor_height]) for path in shortest_path])
            # min_kin_dist = self.path_length * self.robots[0].robot_mass
            # kinematic_disturbance = min_kin_dist / (min_kin_dist + self.kinematic_disturbance)
            # min_dyn_dist = self.current_step * self.robots[0].robot_mass * 9.8
            # dynamic_disturbance_a = min_dyn_dist / (min_dyn_dist + self.dynamic_disturbance_a)
            # dynamic_disturbance_b = self.current_step / float(self.current_step + self.dynamic_disturbance_b)
            # object_files = [obj.filename for obj in self.interactive_objects]
            # episode = Episode(
            #     env=self.scene.model_id,
            #     agent=self.robots[0].model_file,
            #     initial_pos=self.initial_pos,
            #     target_pos=self.target_pos,
            #     geodesic_distance=geodesic_distance,
            #     shortest_path=shortest_path,
            #     agent_trajectory=np.array(self.agent_trajectory),
            #     object_files=object_files,
            #     object_trajectory=np.array(self.object_trajectory),
            #     success=float(info['success']),
            #     path_efficiency=min(1.0, geodesic_distance / self.path_length),
            #     kinematic_disturbance=kinematic_disturbance,
            #     dynamic_disturbance_a=dynamic_disturbance_a,
            #     dynamic_disturbance_b=dynamic_disturbance_b,
            #     collision_step=self.collision_step,
            # )
            # self.stored_episodes.append(episode)

        return done, info

    def get_stored_episodes(self):
        return self.stored_episodes

    def before_simulation(self):
        return None

    def after_simulation(self, cache, collision_links):
        return

    def step(self, action):
        """
        apply robot's action and get state, reward, done and info, following openAI gym's convention

        :param action: a list of control signals
        :return: state: state, reward, done, info
        """
        self.current_step += 1
        self.robots[0].apply_action(action)
        cache = self.before_simulation()
        collision_links = self.run_simulation()
        self.after_simulation(cache, collision_links)

        state = self.get_state(collision_links)
        info = {}
        reward, info = self.get_reward(collision_links, action, info)
        done, info = self.get_termination(collision_links, info)

        if done and self.automatic_reset:
            info['last_observation'] = state
            state = self.reset()
        return state, reward, done, info

    def reset_agent(self):
        max_trials = 100

        for _ in range(max_trials):
            self.robots[0].robot_specific_reset()
            if self.reset_initial_and_target_pos():
                return True
            else:
                return False
        raise Exception("Failed to reset robot without collision")
# =======
#         for _ in range(max_trials):
#             self.robots[0].robot_specific_reset()
#             self.reset_initial_and_target_pos()
#             if self.test_valid_position():
#                 return
#         raise Exception("Failed to reset robot without collision")

    def reset_initial_and_target_pos(self):
        self.robots[0].set_position(pos=self.initial_pos)
        self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(*self.initial_orn), 'wxyz'))
        return True

    def test_valid_position(self):
        collision_links = self.run_simulation()
        collision_links_flatten = [item for sublist in collision_links for item in sublist]
        return len(collision_links_flatten) == 0

    def reset(self):
        """
        Reset the agent to a collision-free start point
        """
        # self.is_resetting = True
        self.current_episode += 1
        agent_reset = False
        self.current_step = 0

        while not agent_reset:
            # print('reset agent!')
            # print('is resetting: {}'.format(self.is_resetting))
            agent_reset = self.reset_agent()
        state = self.get_state()
        # self.is_resetting = False

        if self.reward_type == 'l2':
            self.potential = self.get_l2_potential()
        elif self.reward_type == 'dense':
            self.potential = self.get_potential()

        self.collision_step = 0
        self.kinematic_disturbance = 0.0
        self.dynamic_disturbance_a = 0.0
        self.dynamic_disturbance_b = 0.0
        self.path_length = 0.0
        self.agent_trajectory = []
        self.object_trajectory = []
        self.interactive_objects_collided = set()
        self.energy_cost = 0.0

        # set position for visual objects
        # if self.visual_object_at_initial_target_pos:
        if self.visual_object_at_target_pos:
            # self.initial_pos_vis_obj.set_position(self.initial_pos)
            self.target_pos_vis_obj.set_position(self.current_target_position)

        return state
    
    def remove_all_collisions(self, body):
        for elem in body:
            # Get the number of sub-object/links in the multibody
            for i in range(-1, elem.nbobj):
                p.setCollisionFilterGroupMask(elem.getId(), i, 0, 0)
                p.setCollisionFilterPair(0, elem.getId(), -1, i, 1)


class NavigateRandomEnv(NavigateEnv):
    def __init__(
            self,
            config_file,
            model_id=None,
            mode='headless',
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            automatic_reset=False,
            random_height=False,
            device_idx=0,
    ):
        super(NavigateRandomEnv, self).__init__(config_file,
                                                model_id=model_id,
                                                mode=mode,
                                                action_timestep=action_timestep,
                                                physics_timestep=physics_timestep,
                                                automatic_reset=automatic_reset,
                                                device_idx=device_idx)
        self.random_height = random_height
        self.random_init_z_offset = self.config.get('random_init_z_offset', 0.1)

class NavigateObstaclesEnv(NavigateEnv):
    def __init__(
             self,
             config_file,
             mode='headless',
             action_timestep=1 / 10.0,
             physics_timestep=1 / 240.0,
             automatic_reset=False,
             random_height=False,
             device_idx=0,
    ):
        super(NavigateObstaclesEnv, self).__init__(config_file,
                                                mode=mode,
                                                action_timestep=action_timestep,
                                                physics_timestep=physics_timestep,
                                                automatic_reset=automatic_reset,
                                                device_idx=device_idx)
        self.random_height = random_height

#         self.floor = VisualObject(visual_shape=p.GEOM_BOX, rgba_color=[0.643, 0.643, 0.788, 0.0], half_extents=[20, 20, 0.02], initial_offset=[0, 0, -0.03])
#         self.floor.load()
#         self.floor.set_position([0, 0, 0])
#         self.simulator.import_object(self.floor)
    
#         self.box_poses = [
#             [[np.random.uniform(-2, 1), np.random.uniform(-2, -1), 0], [0, 0, 0, 1]],
#             [[np.random.uniform(2, 1), np.random.uniform(2, -1), 0], [0, 0, 0, 1]],
#             [[np.random.uniform(4, 1), np.random.uniform(-1, -1), 0], [0, 0, 0, 1]]
#             ]
        
        self.box_poses = [
            [[0, -1.5, 0], [0, 0, 0, 1]],
            [[0, 1.5, 0], [0, 0, 0, 1]],
            [[1.5, 0, 0], [0, 0, 0, 1]],
            [[-1.5, 0, 0], [0, 0, 0, 1]]
            ]
    
        self.walls = []
        for box_pose in self.box_poses:
            box = BoxShape(pos=box_pose[0], dim=[0.2, 0.3, 0.3], rgba_color=[1.0, 0.0, 0.0, 1.0])
            self.obstacle_ids.append(self.simulator.import_interactive_object(box))
            self.walls += [box]
        
    def reset_initial_and_target_pos(self):
        floor, pos = self.scene.get_random_point(min_xy=self.initial_pos[0], max_xy=self.initial_pos[1])
        self.robots[0].set_position(pos=[pos[0], pos[1], pos[2] + 0.1])
        self.robots[0].set_orientation(
            orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz'))
        self.initial_pos = pos
        
        max_trials = 100
        dist = 0.0
        for _ in range(max_trials):  # if initial and target positions are < 1 meter away from each other, reinitialize
            _, self.current_target_position = self.scene.get_random_point_floor(floor, min_xy=self.target_pos[0], max_xy=self.target_pos[1], random_height=self.random_height)
            dist = l2_distance(self.initial_pos, self.current_target_position)
#         floor, self.initial_pos = self.scene.get_random_point_floor(self.floor_num, self.random_height)
#         max_trials = 100
        dist = 0.0
        for _ in range(max_trials):  # if initial and target positions are < 1 meter away from each other, reinitialize
            _, self.target_pos = self.scene.get_random_point_floor(self.floor_num, self.random_height)
            dist = l2_distance(self.initial_pos, self.target_pos)
# >>>>>>> master
#             if dist > 1.0:
#                 break
#         if dist < 1.0:
#             raise Exception("Failed to find initial and target pos that are >1m apart")

        collision_links = []
        for _ in range(self.simulator_loop):
            self.simulator_step()
            collision_links.append(list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0])))
        collision_links = self.filter_collision_links(collision_links)
        no_collision = len(collision_links) == 0
        return no_collision
    
class NavigatePedestriansEnv(NavigateEnv):
    def __init__(
             self,
             config_file,
             mode='headless',
             action_timestep=1 / 10.0,
             physics_timestep=1 / 240.0,
             automatic_reset=False,
             random_height=False,
             device_idx=0,
    ):
        super(NavigatePedestriansEnv, self).__init__(config_file,
                                                mode=mode,
                                                action_timestep=action_timestep,
                                                physics_timestep=physics_timestep,
                                                automatic_reset=automatic_reset,
                                                device_idx=device_idx)
        self.random_height = random_height
        self.pedestrian_z = 0.03 # hard-coded.

        self.num_pedestrians = self.config.get('num_pedestrians', 0)
        self.num_obstacles = self.config.get('num_obstacles', 0)
        self.pedestrians_can_see_robot = self.config.get('pedestrians_can_see_robot', False)
        self.randomize_pedestrian_attributes = self.config.get('randomize_pedestrian_attributes', False)                
        
        ''' Walls '''
        # wall = [pos, dim]
        # self.walls = [[[0, 5.25, 0.501], [5, 0.25, 1.5]],
                      # [[0, -5.25, 0.501], [5, 0.25, 1.5]],
                      # [[5, 0, 0.501], [0.25, 5, 1.5]],
                      # [[-5, 0, 0.501], [0.25, 5, 1.5]]]
#                     [[-8.55, 5, 0.501], [1.44, 0.1, 0.5]],
#                     [[8.55, 4, 0.501], [1.44, 0.1, 0.5]],
#                     [[10.2, 5.5 ,0.501],[0.2, 1.5, 0.5]],
#                     [[-10.2, 6, 0.501],[0.2, 1, 0.5]]]
        

        # if layout: # if specify a layout file
            # self.walls = self.layout.get('walls')
            # components = self.layout.get('components')
            # humans = self.layout.get('movements')['humans']
            # agent = self.layout.get('movements')['agent']
            # self.pedestrian_initial_x_ranges = self.layout.get('humans')['initial_x_ranges']
            # self.pedestrian_initial_y_ranges = self.layout.get('humans')['initial_y_ranges']
            # self.pedestrian_target_x_ranges = self.layout.get('humans')['target_x_ranges']
            # self.pedestrian_target_y_ranges = self.layout.get('humans')['target_y_ranges'] 
            # self.min_separation = self.layout.get('humans')['min_separation']
        # else: # use the layout in meta-config
        self.walls = self.config.get('walls')
        components = self.config.get('components')
        self.num_pedestrians = self.config.get('num_pedestrians', 0)
        self.num_obstacles = self.config.get('num_obstacles', 0)
        self.pedestrians_can_see_robot = self.config.get('pedestrians_can_see_robot', False)        

        humans = self.config.get('movements')['humans']
        agent = self.config.get('movements')['agent']

        self.min_separation = self.config.get('humans')['radius'] * 2 + 0.05
        print('min separation: {}'.format(self.min_separation))
        self.min_separation = 0.25
        self.pedestrian_x_range_radius = [humans['x_range_radius'] for _ in range(self.num_pedestrians)]
        self.pedestrian_y_range_radius = [humans['y_range_radius'] for _ in range(self.num_pedestrians)]
        self.flip_prob = humans['flip_prob']
        self.pedestrian_dist_tol = self.dist_tol * 2.0
        # Randomize initial - target positions for each pedestrian.
        self.pedestrian_centers = [[], []]
        initial_target = humans['initial_target']
        for pedestrian in range(self.num_pedestrians):
            initial, target = random.choice(initial_target)
            print('pedestrian: {} initial {} target {}'.format(pedestrian, initial, target))
            self.pedestrian_centers[0].append(components[initial])
            self.pedestrian_centers[1].append(components[target])
        self.pedestrian_status = [False] * self.num_pedestrians 

        self.agent_init = components[agent['initial_positions']]
        self.agent_target = components[agent['target_positions']]
        self.agent_x_range_radius = agent['x_range_radius']
        self.agent_y_range_radius = agent['y_range_radius']

        if self.walls is not None:
            for i, wall_pos in enumerate(self.walls['walls_pos']):
                wall_dim = self.walls['walls_dim'][i]
<<<<<<< HEAD
                box = BoxShape(pos=wall_pos, dim=wall_dim)
=======
                box = BoxShape(pos=wall_pos, dim=wall_dim, mass=10)
>>>>>>> c958d0899b63bb8de79920ed4084f407ca78a825
                self.obstacle_ids.append(self.simulator.import_object(box))
        print('=' * 100)
        print('WALLS IDS: {}'.format(self.obstacle_ids))
        print('SCENE IDS: {}'.format(self.scene_ids))
        print('ROBOT IDS: {}'.format(self.robots[0].robot_ids[0]))
        
#         for body_id in self.collision_ignore_link_a_ids:
#             p.setCollisionFilterGroupMask(body_id, -1, 0, 0)
#             p.setCollisionFilterPair(0, body_id, -1, -1, 0)

        ''' Obstacles '''
        self.obstacles = []

        self.initial_box_size = [0.2, 0.3, 0.3]
        # Create/re-create obstacles in random positions
        self.reset_obstacles() 
        
        ''' Pedestrians '''
        self.humans = []
        self.pedestrians = []
        self.pedestrian_ids = []
        self.pedestrian_goal_ids = []
        self.pedestrian_goal_objects = []
        self.pedestrian_gibson_ids = []

        # poses are defined as x,y,z followed by random lower and upper range to add to pose
        # self.pedestrian_start_poses = [[(3.0, 3.0, 0.03), (-1.0, 1.0)], [(2.0, 2.0, 0.03), (-1.0, 1.0)],
                                       # [(-3.0, -3.0, 0.03), (-1.0, 1.0)], [(-2.0, -2.0, 0.03), (-1.0, 1.0)]]
        # self.pedestrian_goal_poses = [[(3.0, 3.0, 0.03), (-1.0, 1.0)], [(2.0, 2.0, 0.03), (-1.0, 1.0)],
                                      # [(-3.0, -3.0, 0.03), (-1.0, 1.0)], [(-2.0, -2.0, 0.03), (-1.0, 1.0)]]
        
        self.reset_pedestrians()

        # Visualize locations of different compoenents to debug.
        for component in components:
            component_pos = components[component]
            p.addUserDebugText(component, [component_pos['center_x'], component_pos['center_y'], 2.0], [1, 0, 0])
        
    def create_pedestrians(self, pedestrian_poses):
#         # remove any existing pedestrians
#         for pedestrian in self.pedestrians:
#             p.removeCollisionShape(pedestrian.collision_id)
#             #self.remove_all_collisions(pedestrian.body_id)            
#             p.removeConstraint(pedestrian.cid)
#             
#         for ped_id in self.pedestrian_gibson_ids:
#             try:
#                 p.removeBody(ped_id)
#             except:
#                 print("ERROR with PEDESTRIAN ID: ", ped_id)
#         
#         self.pedestrian_gibson_ids = []

        # create pedestrian objects
        if len(self.pedestrians) == 0:
            self.pedestrians = [Pedestrian(pos = pedestrian_poses[i]) for i in range(self.num_pedestrians)]
            # spawn pedestrians and get Gibson IDs
            self.pedestrian_gibson_ids = [self.simulator.import_object(ped) for ped in self.pedestrians]
<<<<<<< HEAD
=======

#         # disable collision checking between pedestrians and floor which speeds up simulation 5x
#         for pedestrian_id in self.pedestrian_gibson_ids:
#             self.collision_ignore_body_b_ids = set(self.config.get('collision_ignore_body_b_ids', self.scene_ids))
#             self.collision_ignore_link_a_ids = set(self.config.get('collision_ignore_link_a_ids', []))
# 
#             for ignore_body_id in self.collision_ignore_body_b_ids:
#                 for ignore_link_id in self.collision_ignore_link_a_ids:                
#                     p.setCollisionFilterPair(pedestrian_id, ignore_body_id, ignore_link_id, 0, 0)
>>>>>>> c958d0899b63bb8de79920ed4084f407ca78a825
    
    def step(self, action):
        # compute the next human actions from the current observations
        # print('self.is_resetting: {}'.format(self.is_resetting))
        human_actions = []
        # print('step is called!!!')
        
        for i, human in enumerate(self.humans):
            # get the positions and velocities of other humans
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
                      
            #self.pedestrians_can_see_robot = np.random.uniform()
            #if self.pedestrians_can_see_robot < 0.5:
            
            # can this person see the robot?
            if self.pedestrians_can_see_robot:
                ob += [self.get_robot_observable_state()]
                #self.pedestrian_simulator.setAgentPosition(self.robot_as_pedestrian_id, tuple(self.robots[0].get_position()[:2]))

            if self.walls:
                walls_config = list(zip(self.walls['walls_pos'], self.walls['walls_dim']))
            else:
                walls_config = list()
                
            human_actions.append(human.act(ob, walls=walls_config, obstacles=self.obstacles))

            # Set next goal if close to current one.
            if l2_distance(np.array(human.get_goal_position()), self.pedestrians[i].get_position()) < self.pedestrian_dist_tol:
                prob = random.random()
                if prob < self.flip_prob:
                    self.reset_single_pedestrian(i)
                
        # move each human
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
            
        # move the robot
        self.robots[0].apply_action(action)
        
        # check for collisions
        collision_links = self.run_simulation()
        
        # get the next state
        state = self.get_state(collision_links)

        # import matplotlib.pyplot as plt
        # print('=' * 100)
        # print('2D: {}'.format(state['sensor']))
        # print('DEPTH: {}'.format(state['depth'][:, :, 0]))
        # print('DEPTH shape: {}'.format(state['depth'].shape))
        # plt.imshow(state['depth'][:, :, 0], cmap='gray')
        # plt.show()
        # xit(0)
        # print('RGB: {}'.format(state['rgb']))
        # print('=' * 100)
        
        # collect reward
        info = {}
        # if not self.is_resetting:
        reward, info = self.get_reward(collision_links, action, info)
        # check for a termination result
        self.current_step += 1
        done, info = self.get_termination(collision_links, info)

        # else:
            # print('REWARD: {}'.format(reward))
            # print('DONE: {}'.format(done))

        # Update distance metrics
        self.time_elapsed += self.action_timestep

        robot_position = self.robots[0].get_position()
        distance_traveled = np.sqrt((robot_position[0] - self.last_robot_px)**2 + (robot_position[1] - self.last_robot_py)**2)
        
        self.distance_traveled += distance_traveled
        self.episode_distance += distance_traveled
        
        #self.n_personal_space_violations += 1 * (self.discomfort_dist - dmin) / self.discomfort_dist * self.time_step
                
        self.last_robot_px = robot_position[0]
        self.last_robot_py = robot_position[1]
        

        if done:
            # print('Episodes: {}\n--Return:{}\n--Success Rate: {}\n--COLLISION RATE: {}\n--TIMEOUT RATE: {}\n--PEDESTRIAN HIT ROBOT RATE:{}\n'.format())
            print("Episodes:", self.current_episode, [(key, np.around(info[key][0], 2)) for key in ['success_rate', 'ped_collision_rate', 'ped_hits_robot_rate', 'collision_rate', 'timeout_rate', 'personal_space_violations', 'shortest_path_length']])
            # TODO: add return here?
            if self.automatic_reset:
                info['last_observation'] = state
                state = self.reset()
            print('=' * 100)
                    
        return state, reward, done, info

    def get_reward(self, collision_links=[], action=None, info={}):
        reward = 0.0
        self.success = False
        self.collision = False
            
        # is the goal reached?
        if l2_distance(self.current_target_position, self.get_position_of_interest()) < self.dist_tol:
            reward = self.success_reward  # |success_reward| = 10.0 per step
            self.success = True
        else:
            # check for collision
            collision_reward = 0
            for collision_link in collision_links:
              if collision_link != []:
                collision_reward += 1
            # collision_reward = float(len(collision_links) > 0)
            if collision_reward > 0:
                self.collision = True
                
                # only blame rhe robot if it is moving at an appreciable speed
                robot_velocity = self.robots[0].get_velocity()
    
                if np.linalg.norm([robot_velocity[0], robot_velocity[1]]) > 0.05:
                    info['collision_reward'] = collision_reward * self.collision_reward_weight  # expose collision reward to info
                    reward = collision_reward * self.collision_reward_weight  # |collision_reward| ~= 1.0 per step if collision
            else:
                # add up the other rewards
                reward = self.slack_reward  # |slack_reward| = 0.01 per step

                if self.current_step == 0:
                    self.initial_potential = self.get_potential()
                    self.normalized_potential = 1.0
                
                current_potential = self.get_potential()
                new_normalized_potential = current_potential / self.initial_potential                
                potential_reward = self.normalized_potential - new_normalized_potential
                
                reward += potential_reward * self.potential_reward_weight
                #print(self.initial_potential, current_potential, self.normalized_potential, new_normalized_potential, potential_reward)
                self.normalized_potential = new_normalized_potential        
        
                # electricity_reward = np.abs(self.robots[0].joint_speeds * self.robots[0].joint_torque).mean().item()
                electricity_reward = 0.0
                reward += electricity_reward * self.electricity_reward_weight  # |electricity_reward| ~= 0.05 per step
        
                # stall_torque_reward = np.square(self.robots[0].joint_torque).mean()
                stall_torque_reward = 0.0
                reward += stall_torque_reward * self.stall_torque_reward_weight  # |stall_torque_reward| ~= 0.05 per step
                
        # # get minimum distance between the robot and a pedestrian
        # ped_positions = get_ped_positions(self)
        # robot_position = self.robots[0].get_position()

        # # TODO: human and robot radius hard coded for now (0.3 meters each)
        # self.collision = False
        # ped_robot_distances = [[np.linalg.norm(ped_pos[i][0] - robot_pos[0], ped_pos[i][1] - robot_pos[1]) - 0.3 - 0.3] for i in range(self.num_pedestrians)]
        # closest_dist = np.min(ped_robot_distances)

        # # velocity projection to elongate the personal space in the direction of motion
        # cutting_off = False

        # ped_robot_lookahead_distances = [(self.humans[i].vx, self.humans[i].vy) for i in range(self.um_pedestrians)]        
        
        # for i, human in enumerate(self.humans):
        #     px = human.px - self.robot.px
        #     py = human.py - self.robot.py

        #     ex = px + human.vx * self.lookahead_interval
        #     ey = py + human.vy * self.lookahead_interval
            
        #     # get the nearest distance to segment connecting the current position and future position
        #     velocity_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius

        #     if velocity_dist < velocity_dmin:
        #         velocity_dmin = velocity_dist
            
        info['episodes'] = self.current_episode
        info['successes'] = self.n_successes,
        info['collisions'] = self.n_collisions,
        info['ped_collisions'] = self.n_ped_collisions,
        info['ped_hits_robot'] = self.n_ped_hits_robot,
        info['timeouts'] = self.n_timeouts,
        info['personal_space_violations'] = 0 if self.distance_traveled == 0 else self.n_personal_space_violations / self.distance_traveled,
        info['cutting_off'] = 0 if self.distance_traveled == 0 else self.n_cutting_off / self.distance_traveled,
        info['success_rate'] = 0 if self.current_episode == 0 else 100 * self.n_successes / self.current_episode,
        info['collision_rate'] = 0 if self.current_episode == 0 else 100 * self.n_collisions / self.current_episode,
        info['ped_collision_rate'] = 0 if self.current_episode == 0 else 100 * self.n_ped_collisions / self.current_episode,
        info['ped_hits_robot_rate'] = 0 if self.current_episode == 0 else 100 * self.n_ped_hits_robot / self.current_episode,
        info['timeout_rate'] = 0 if self.current_episode == 0 else 100 * self.n_timeouts / self.current_episode,
        info['shortest_path_length'] = None if self.current_episode == 0 else [self.spl]
        
        if self.success:
            print(reward)
                         
        return reward, info

    def update_pedestrian_goal_markers(self, pedestrian_goals):
        if len(self.pedestrian_goal_objects) == 0:
            for ped_goal in pedestrian_goals:
                pedestrian_goal_visual_obj = VisualObject(visual_shape=p.GEOM_CYLINDER,
                                                          rgba_color=[1, 1, 0, 0.6],
                                                          radius=0.05,
                                                          length=0.5,
                                                          initial_offset=[0, 0, 0.25])

                self.pedestrian_goal_objects.append(pedestrian_goal_visual_obj)
                self.pedestrian_goal_ids.append(pedestrian_goal_visual_obj.load())                
        # else:
        for i in range(len(self.pedestrian_goal_objects)):
            self.pedestrian_goal_objects[i].set_position(pos=[pedestrian_goals[i][0], pedestrian_goals[i][1], 0.25])
        


    def create_human_agents(self):
        for i in range(self.num_pedestrians):
            human = Human(self.config, 'humans', num_pedestrians=self.num_pedestrians)
            self.humans.append(human)
            if self.randomize_pedestrian_attributes:
                human.sample_random_attributes()

    def reset_single_pedestrian(self, pid):
        # Randomize goal.
        status = self.pedestrian_status[pid]
        center_x = self.pedestrian_centers[status][pid]['center_x']
        center_y = self.pedestrian_centers[status][pid]['center_y']
        gx = random.uniform(center_x - self.pedestrian_x_range_radius[pid], center_x + self.pedestrian_x_range_radius[pid])
        gy = random.uniform(center_y - self.pedestrian_y_range_radius[pid], center_y + self.pedestrian_y_range_radius[pid])
        
        # Reset Human Agent.
        px, py, theta = self.humans[pid].get_position()
        self.humans[pid].set(px, py, theta, gx, gy, gr=0, vx=0, vy=0, vr=0)

        # Reset goal marker.
        self.pedestrian_goal_objects[pid].set_position(pos=[gx, gy, 0.03])

        # Flip status.
        self.pedestrian_status[pid] = not self.pedestrian_status[pid]

    def reset_pedestrians(self):
        if len(self.pedestrians) == 0:
            # create pedestrians and generate initial poses
            pedestrian_poses = self.generate_pedestrian_poses_v2()
            self.create_pedestrians(pedestrian_poses)
            self.create_human_agents()
        else:
            # get current poses
            pedestrian_poses = [pedestrian.get_position() for pedestrian in self.pedestrians]
            
        # generate a goal for each pedestrian
        pedestrian_goals = self.generate_pedestrian_poses_v2()

        # generate the goal marker in Gibson
        self.update_pedestrian_goal_markers(pedestrian_goals)
        
        self.humans = []
        for i in range(self.num_pedestrians):
            human = Human(self.config, 'humans')
            
            if self.randomize_pedestrian_attributes:
                self.humans[i].sample_random_attributes()
                
            [px, py, _] = pedestrian_poses[i]
            [gx, gy, _] = pedestrian_goals[i]
 
            vx = human.v_pref * np.cos(human.theta)
            vy = human.v_pref * np.sin(human.theta)
            human.set(px, py, human.theta, gx, gy, vx, vy, 0, 0)
            
            self.humans.append(human)
    
            self.humans[i].set(px, py, self.humans[i].theta, gx, gy, 0, 0, 0, 0)

    def update_pedestrian_positions_in_gibson(self):
        for i, human in enumerate(self.humans):
            
            if human.reached_destination():
                continue
            
            px = human.px
            py = human.py
            theta = human.theta
        
            direction = p.getQuaternionFromEuler([0, 0, theta])
            self.pedestrians[i].reset_position_orientation([px, py, 0.03], direction)        
            
    def reset_obstacles(self):
        # First remove the obstacles
        # print('RESET {} OBSTACLES'.format(self.num_obstacles))
        for obstacle in self.obstacles:
            p.removeCollisionShape(obstacle.collision_id)
            p.removeBody(obstacle.body_id)

        self.obstacle_ids = list()
        self.obstacles = list()

        # Then recreate them in new positions
        for _ in range(self.num_obstacles):
            _, pos = self.scene.get_random_point(min_xy=-2, max_xy=2)            
            box = BoxShape(pos=pos, dim=self.initial_box_size, rgba_color=[1.0, 0.0, 0.0, 1.0])
            self.obstacle_ids.append(self.simulator.import_interactive_object(box))
            self.obstacles.append(box)
        
    def reset_initial_and_target_pos(self):
        # Compute SPL metric
        robot_position = self.robots[0].get_position()
        self.last_robot_px = robot_position[0]
        self.last_robot_py = robot_position[1]
        
        if self.current_episode > 0:
            self.compute_metrics()
            
        # Select new positions for obstacles
        # self.reset_obstacles()

        # Select new positions and goals for pedestrians
        # self.reset_pedestrians()
        
        # Choose new start postion and goal location for robot
        reset_complete = False
        while not reset_complete:
            pos = np.array([np.random.uniform(self.agent_init['center_x'] - self.agent_x_range_radius, self.agent_init['center_x'] + self.agent_x_range_radius), np.random.uniform(self.agent_init['center_y'] - self.agent_y_range_radius, self.agent_init['center_y'] + self.agent_y_range_radius), 0.0])

            self.robots[0].set_position(pos=[pos[0], pos[1], pos[2] + 0.1])
            self.robots[0].set_orientation(
                orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz'))
            self.initial_pos = np.array(pos)
            
            max_trials = 1000
            dist = 0.0
            for _ in range(max_trials):  # if initial and target positions are < 1 meter away from each other, reinitialize
                self.current_target_position = np.array([np.random.uniform(self.agent_target['center_x'] - self.agent_x_range_radius, self.agent_target['center_x'] + self.agent_x_range_radius), np.random.uniform(self.agent_target['center_y']- self.agent_y_range_radius, self.agent_target['center_y'] + self.agent_y_range_radius), 0.0])
                dist = l2_distance(self.initial_pos, self.current_target_position)
                if dist > 1.0:
                    reset_complete = True
                    break
            if dist < 1.0:
                print("Failed to find initial and target pos that are >1m apart")
                print("Selecting new initial position")
        
        self.n_steps = 0
        self.episode_distance = 0.0
        self.episode_time = 0.0
        
#         # Get initial goal potential and collision potential
#         if self.n_steps == 0:
#             self.initial_potential = self.get_potential()
#             self.normalized_potential = 1.0

        collision_links = []
        for _ in range(self.simulator_loop):
            self.simulator_step()
            collision_links.append(list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0])))
        collision_links = self.filter_collision_links(collision_links)
        # no_collision = len(collision_links) == 0
        no_collision = True
        for collision_link in collision_links:
            if collision_link != []:
                no_collision = False

        # Compute the shortest distance to this goal (TODO: account for obstacles!)
        robot_position = self.robots[0].get_position()

        self.episode_shortest_distance = np.sqrt((self.current_target_position[0] - robot_position[0])**2 + (self.current_target_position[1] - robot_position[1])**2)
        
        self.n_steps = 0
        self.episode_distance = 0.0
        self.episode_time = 0.0

        return no_collision
        # return True
    
    def compute_metrics(self):
        if self.success:
            self.spl_sum += self.episode_shortest_distance / (max(self.episode_shortest_distance, self.episode_distance))
        
        self.spl = self.spl_sum / self.current_episode
        
    def generate_pedestrian_poses_v2(self):
        # Euclidean distance on xy plane.
        def euclidean_xy(source, target):
            return np.sqrt((source[0] - target[0]) ** 2 + (source[1] - target[1]) ** 2)

        pedestrian_poses = list()
        all_object_poses = list()
       
        for obstacle in self.obstacles:
            all_object_poses.append(obstacle.get_position())

        for i in range(self.num_pedestrians):
            good_pose = False
            while not good_pose:
                good_pose = True
                status = int(self.pedestrian_status[i])
                center_x = self.pedestrian_centers[status][i]['center_x']
                pedestrian_x = random.uniform(center_x - self.pedestrian_x_range_radius[i], center_x + self.pedestrian_x_range_radius[i])
                center_y = self.pedestrian_centers[status][i]['center_y']
                pedestrian_y = random.uniform(center_y - self.pedestrian_y_range_radius[i], center_y + self.pedestrian_y_range_radius[i])
                pedestrian_pose = (pedestrian_x, pedestrian_y, self.pedestrian_z)
                
                # Check if this position is too closed to any previous poses.
                for prev_pose in all_object_poses:
                    dist = euclidean_xy(pedestrian_pose, prev_pose)
                    # print('distance between instances: {}'.format(dist))
                    if dist < self.min_separation:
                        good_pose = False
                        
            # Flip status
            self.pedestrian_status[i] = not self.pedestrian_status[i]
    
            pedestrian_poses.append(pedestrian_pose)
            all_object_poses.append(pedestrian_pose)

        print('pedestrian poses: {}'.format(pedestrian_poses))
        return pedestrian_poses

    def get_robot_observable_state(self):
        px, py, pz = self.robots[0].get_position()
        theta = self.robots[0].get_rpy()[2]
        vx, vy, vz = self.robots[0].get_velocity()
        vr = self.robots[0].get_angular_velocity()[2]

        # TODO: replace hard coded robot radius and personal space radius
        return ObservableState(px, py, theta, vx, vy, vr, 0.3, 0.6)

class InteractiveGibsonNavigateEnv(NavigateRandomEnv):
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
        super(InteractiveGibsonNavigateEnv, self).__init__(config_file,
                                                           model_id=model_id,
                                                           mode=mode,
                                                           action_timestep=action_timestep,
                                                           physics_timestep=physics_timestep,
                                                           automatic_reset=automatic_reset,
                                                           random_height=False,
                                                           device_idx=device_idx)
        self.collision_reward_weight = collision_reward_weight

        self.replaced_objects = []
        self.replaced_objects_pos = []
        self.additional_objects = []
        self.class_map = {}

        self.should_load_replaced_objects = self.config.get('should_load_replaced_objects', False)
        self.should_load_additional_objects = self.config.get('should_load_additional_objects', False)

        self.build_class_map()
        self.load_replaced_objects()
        self.load_additional_objects()
        self.interactive_objects = self.replaced_objects + self.additional_objects
        self.new_potential = None

        self.visualize_waypoints = True
        if self.visualize_waypoints and self.mode == 'gui':
            cyl_length = 0.2
            self.waypoints_vis = [VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                               rgba_color=[0, 1, 0, 0.3],
                                               radius=0.1,
                                               length=cyl_length,
                                               initial_offset=[0, 0, cyl_length / 2.0]) for _ in range(10)]
            for waypoint in self.waypoints_vis:
                waypoint.load()

    def build_class_map(self):
        # 0 and 1 are reserved for mesh and robot
        init_class_id = 2

        if self.should_load_replaced_objects:
            self.replaced_object_classes = [
                '03001627',  # chair
                '04256520',  # couch
                '04379243',  # table / desk
                '00000001',  # door
            ]
            for item in self.replaced_object_classes:
                self.class_map[item] = init_class_id
                init_class_id += 1

        if self.should_load_additional_objects:
            self.additional_objects_path = [
                'object_2eZY2JqYPQE.urdf',
                'object_lGzQi2Pk5uC.urdf',
                'object_ZU6u5fvE8Z1.urdf',
                'object_H3ygj6efM8V.urdf',
                'object_RcqC01G24pR.urdf'
            ]

            for item in self.additional_objects_path:
                self.class_map[item] = init_class_id
                init_class_id += 1

        self.num_object_classes = init_class_id

    def load_replaced_objects(self):
        if not self.should_load_replaced_objects:
            return
        scene_path = os.path.join(gibson2.assets_path, 'dataset', self.scene.model_id)
        urdf_files = [item for item in os.listdir(scene_path) if item[-4:] == 'urdf']
        position_files = [item[:-4].replace('alignment_centered', 'pos') + 'txt' for item in urdf_files]

        for urdf_file, position_file in zip(urdf_files, position_files):
            with open(os.path.join(scene_path, position_file)) as f:
                pos = np.array([float(item) for item in f.readlines()[0].strip().split()])
                # filter out duplicate annotations for the same object
                if len(self.replaced_objects_pos) == 0 or \
                        np.min(np.linalg.norm(np.array(self.replaced_objects_pos) - pos, axis=1)) > 0.5:
                    class_id = urdf_file.split('.')[0].split('_')[-1]
                    obj = InteractiveObj(os.path.join(scene_path, urdf_file))
                    self.simulator.import_object(obj, class_id=self.class_map[class_id])
                    self.replaced_objects.append(obj)
                    self.replaced_objects_pos.append(pos)

    def load_additional_objects(self):
        if not self.should_load_additional_objects:
            return
        num_dupicates = 2
        for _ in range(num_dupicates):
            for urdf_model in self.additional_objects_path:
                obj = InteractiveObj(os.path.join(gibson2.assets_path, 'models/sample_urdfs', urdf_model))
                self.simulator.import_object(obj, class_id=self.class_map[urdf_model])
                self.additional_objects.append(obj)

    def global_to_local(self, pos):
        return rotate_vector_3d(pos - self.robots[0].get_position(), *self.robots[0].get_rpy())

    def get_additional_states(self):
        target_pos_local = self.global_to_local(self.target_pos)
        linear_velocity_local = rotate_vector_3d(self.robots[0].robot_body.velocity(),
                                                 *self.robots[0].get_rpy())
        angular_velocity_local = rotate_vector_3d(self.robots[0].robot_body.angular_velocity(),
                                                  *self.robots[0].get_rpy())

        source = self.robots[0].get_position()[:2]
        target = self.target_pos[:2]
        shortest_path, geodesic_dist = self.scene.get_shortest_path(self.floor_num, source, target)

        robot_z = self.robots[0].get_position()[2]
        if self.visualize_waypoints and self.mode == 'gui':
            for i in range(10):
                self.waypoints_vis[i].set_position(pos=np.array([shortest_path[i][0], shortest_path[i][1], robot_z]))
        waypoints_local_xy = np.array([self.global_to_local(np.concatenate((waypoint, [robot_z])))[:2]
                                       for waypoint in shortest_path]).flatten()
        additional_states = np.concatenate((waypoints_local_xy,
                                            target_pos_local,
                                            linear_velocity_local,
                                            angular_velocity_local))
        # cache results for reward calculation
        self.new_potential = geodesic_dist

        assert len(additional_states) == self.additional_states_dim, 'additional states dimension mismatch'
        return additional_states

    def get_potential(self):
        return self.new_potential

    def reset_additional_objects(self):
        for obj in self.additional_objects:
            while True:
                _, pos = self.scene.get_random_point_floor(self.floor_num, self.random_height)
                obj.set_position_rotation([pos[0], pos[1], pos[2] + self.random_init_z_offset],
                                          quatToXYZW(euler2quat(0.0, 0.0, 0.0), 'wxyz'))
                has_collision = False
                for _ in range(self.simulator_loop):
                    self.simulator_step()
                    if len(p.getContactPoints(bodyA=obj.body_id)) > 0:
                        has_collision = True
                        break
                if not has_collision:
                    break

    def reset_replaced_objects(self):
        for obj, pos in zip(self.replaced_objects, self.replaced_objects_pos):
            obj.set_position_rotation([pos[0], pos[1], pos[2] + self.random_init_z_offset],
                                      quatToXYZW(euler2quat(0.0, 0.0, 0.0), 'wxyz'))

    def reset(self):
        self.floor_num = self.scene.get_random_floor()
        self.scene.reset_floor(floor=self.floor_num, additional_elevation=0.05)
        self.reset_replaced_objects()
        self.reset_additional_objects()

        self.new_potential = None
        state = NavigateEnv.reset(self)
        return state

    def before_simulation(self):
        robot_position = self.robots[0].get_position()
        object_positions = [obj.get_position() for obj in self.interactive_objects]
        return robot_position, object_positions

    def after_simulation(self, cache, collision_links):
        robot_position, object_positions = cache

        collision_links_flatten = [item for sublist in collision_links for item in sublist]
        if len(collision_links_flatten) > 0:
            self.dynamic_disturbance_a += np.mean([
                np.linalg.norm(
                    np.sum([elem[9] * np.array(elem[7]) for elem in sublist], axis=0)  # sum of all forces
                )
                for sublist in collision_links])
            collision_objects = set([col[2] for col in collision_links_flatten])
            self.dynamic_disturbance_b += len(collision_objects)
            self.interactive_objects_collided |= collision_objects

        self.agent_trajectory.append(np.concatenate((self.robots[0].get_position(), self.robots[0].get_orientation())))
        self.object_trajectory.append([np.concatenate((obj.get_position(), obj.get_orientation()))
                                       for obj in self.interactive_objects])
        self.path_length += np.linalg.norm(self.robots[0].get_position() - robot_position)
        self.kinematic_disturbance += np.sum([
            obj.mass * np.linalg.norm(np.array(obj.get_position()) - np.array(prev_pos))
            for obj, prev_pos in zip(self.interactive_objects, object_positions)
            if obj.body_id in self.interactive_objects_collided
        ])

        self.robots[0].set_position(pos=[pos[0], pos[1], pos[2] + 0.1])
        self.robots[0].set_orientation(
            orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz'))
        self.initial_pos = np.array(pos)
        
        max_trials = 1000
        dist = 0.0
        for _ in range(max_trials):  # if initial and target positions are < 1 meter away from each other, reinitialize
            self.current_target_position = np.array([np.random.uniform(self.agent_target['center_x'] - self.agent_x_range_radius, self.agent_target['center_x'] + self.agent_x_range_radius), np.random.uniform(self.agent_target['center_y']- self.agent_y_range_radius, self.agent_target['center_y'] + self.agent_y_range_radius), 0.0])
            dist = l2_distance(self.initial_pos, self.current_target_position)
            if dist > 1.0:
                reset_complete = True
                break
        if dist < 1.0:
            print("Failed to find initial and target pos that are >1m apart")
            print("Selecting new initial position")
        
        self.n_steps = 0
        self.episode_distance = 0.0
        self.episode_time = 0.0
        
#         # Get initial goal potential and collision potential
#         if self.n_steps == 0:
#             self.initial_potential = self.get_potential()
#             self.normalized_potential = 1.0

        collision_links = []
        for _ in range(self.simulator_loop):
            self.simulator_step()
            collision_links.append(list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0])))
        collision_links = self.filter_collision_links(collision_links)
        no_collision = len(collision_links) == 0

        # Compute the shortest distance to this goal (TODO: account for obstacles!)
        robot_position = self.robots[0].get_position()

        self.episode_shortest_distance = np.sqrt((self.current_target_position[0] - robot_position[0])**2 + (self.current_target_position[1] - robot_position[1])**2)
        
        self.n_steps = 0
        self.episode_distance = 0.0
        self.episode_time = 0.0

        #return no_collision
        return True

class InteractiveNavigateEnv(NavigateEnv):
    def __init__(self,
                 config_file,
                 mode='headless',
                 action_timestep=1 / 10.0,
                 physics_timestep=1 / 240.0,
                 random_position=False,
                 device_idx=0,
                 automatic_reset=False,
                 arena="simple_hl_ll"):
        super(InteractiveNavigateEnv, self).__init__(config_file,
                                                     mode=mode,
                                                     action_timestep=action_timestep,
                                                     physics_timestep=physics_timestep,
                                                     automatic_reset=automatic_reset,
                                                     device_idx=device_idx)
        self.arena = arena
        assert self.arena in [
            "only_ll_obstacles",
            "only_ll",
            "simple_hl_ll",
            "complex_hl_ll"
        ], "Wrong arena"

        self.floor = VisualMarker(visual_shape=p.GEOM_BOX, rgba_color=[0.643, 0.643, 0.788, 0.0],
                                  half_extents=[20, 20, 0.02], initial_offset=[0, 0, -0.03])
        self.floor.load()
        self.floor.set_position([0, 0, 0])
        self.simulator.import_object(self.floor, class_id=0)

        self.door = InteractiveObj(os.path.join(gibson2.assets_path, 'models', 'scene_components', 'realdoor.urdf'),
                                   scale=1.35)
        self.simulator.import_interactive_object(self.door, class_id=2)
        if self.arena == "only_ll" or self.arena == "only_ll_obstacles":
            self.door.set_position_rotation([100.0, 100.0, -0.03], quatToXYZW(euler2quat(0, 0, np.pi / 2.0), 'wxyz'))
        else:
            self.door.set_position_rotation([0.0, 0.0, -0.03], quatToXYZW(euler2quat(0, 0, -np.pi / 2.0), 'wxyz'))
        self.door_angle = self.config.get('door_angle', 90)
        self.door_angle = (self.door_angle / 180.0) * np.pi
        self.door_handle_link_id = 2
        self.door_axis_link_id = 1
        self.jr_end_effector_link_id = 33  # 'm1n6s200_end_effector'
        self.random_position = random_position

        if self.arena == "only_ll_obstacles":
            self.box_poses = [
                [[np.random.uniform(-4, 4), np.random.uniform(-4, -1), 1], [0, 0, 0, 1]],
                [[np.random.uniform(-4, 4), np.random.uniform(-4, -1), 1], [0, 0, 0, 1]],
                [[np.random.uniform(-4, 4), np.random.uniform(-4, -1), 1], [0, 0, 0, 1]],

                [[np.random.uniform(-4, 4), np.random.uniform(1, 4), 1], [0, 0, 0, 1]],
                [[np.random.uniform(-4, 4), np.random.uniform(1, 4), 1], [0, 0, 0, 1]],
                [[np.random.uniform(-4, 4), np.random.uniform(1, 4), 1], [0, 0, 0, 1]],

                [[np.random.uniform(-4, -1), np.random.uniform(-4, 4), 1], [0, 0, 0, 1]],
                [[np.random.uniform(-4, -1), np.random.uniform(-4, 4), 1], [0, 0, 0, 1]],
                [[np.random.uniform(-4, -1), np.random.uniform(-4, 4), 1], [0, 0, 0, 1]],

                [[np.random.uniform(1, 4), np.random.uniform(-4, 4), 1], [0, 0, 0, 1]],
                [[np.random.uniform(1, 4), np.random.uniform(-4, 4), 1], [0, 0, 0, 1]],
                [[np.random.uniform(1, 4), np.random.uniform(-4, 4), 1], [0, 0, 0, 1]],
            ]

            self.walls = []
            for box_pose in self.box_poses:
                box = BoxShape(pos=box_pose[0], dim=[0.5, 0.5, 1])
                self.simulator.import_object(box, class_id=3)
                self.walls += [box]

        elif self.arena == "only_ll":
            self.wall_poses = [
                [[0, -3, 1], [0, 0, 0, 1]],
                [[0, 3, 1], [0, 0, 0, 1]],
                [[-3, 0, 1], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]],
                [[3, 0, 1], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]],
            ]

            self.walls = []
            for wall_pose in self.wall_poses:
                wall = InteractiveObj(os.path.join(gibson2.assets_path, 'models', 'scene_components', 'walls.urdf'),
                                      scale=1)
                self.simulator.import_object(wall, class_id=3)
                wall.set_position_rotation(wall_pose[0], wall_pose[1])
                self.walls += [wall]

        elif self.arena == "simple_hl_ll":
            self.wall_poses = [
                [[0, -3, 1], [0, 0, 0, 1]],
                [[0, 3, 1], [0, 0, 0, 1]],
                [[-3, 0, 1], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]],
                [[3, 0, 1], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]],
                [[0, -7.8, 1], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]],
                [[0, 7.8, 1], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]],
            ]

            self.walls = []
            for wall_pose in self.wall_poses:
                wall = InteractiveObj(os.path.join(gibson2.assets_path, 'models', 'scene_components', 'walls.urdf'),
                                      scale=1)
                self.simulator.import_object(wall, class_id=3)
                wall.set_position_rotation(wall_pose[0], wall_pose[1])
                self.walls += [wall]

        elif self.arena == "complex_hl_ll":
            self.wall_poses = [
                [[0, -3, 1], [0, 0, 0, 1]],
                [[0, 6, 1], [0, 0, 0, 1]],
                [[-3, 0, 1], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]],
                [[3, 0, 1], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]]
            ]

            self.half_wall_poses = [
                [[1.3, 3, 1], [0, 0, 0, 1]],
            ]

            self.quarter_wall_poses = [
                [[0.0, 7.68, 1], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]],
                [[0.0, 2.45, 1], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]],
            ]

            self.walls = []
            for wall_pose in self.wall_poses:
                wall = InteractiveObj(os.path.join(gibson2.assets_path, 'models', 'scene_components', 'walls.urdf'),
                                      scale=1)
                self.simulator.import_object(wall, class_id=3)
                wall.set_position_rotation(wall_pose[0], wall_pose[1])
                self.walls += [wall]

            for wall_pose in self.half_wall_poses:
                wall = InteractiveObj(
                    os.path.join(gibson2.assets_path, 'models', 'scene_components', 'walls_half.urdf'),
                    scale=1)
                self.simulator.import_object(wall, class_id=3)
                wall.set_position_rotation(wall_pose[0], wall_pose[1])
                self.walls += [wall]

            for wall_pose in self.quarter_wall_poses:
                wall = InteractiveObj(
                    os.path.join(gibson2.assets_path, 'models', 'scene_components', 'walls_quarter.urdf'),
                    scale=1)
                self.simulator.import_object(wall, class_id=3)
                wall.set_position_rotation(wall_pose[0], wall_pose[1])
                self.walls += [wall]

        # dense reward
        self.prev_stage = self.stage
        self.stage_get_to_door_handle = 0
        self.stage_open_door = 1
        self.stage_get_to_target_pos = 2

        # attaching JR's arm to the door handle
        self.door_handle_dist_thresh = 0.2
        self.cid = None

        # visualize subgoal
        cyl_length = 3.0
        self.subgoal_end_effector = VisualMarker(rgba_color=[0, 0, 0, 0.8], radius=0.06)
        self.subgoal_end_effector.load()
        self.subgoal_end_effector_base = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                                      rgba_color=[1, 1, 0, 0.8],
                                                      radius=0.05,
                                                      length=cyl_length,
                                                      initial_offset=[0, 0, cyl_length / 2])
        self.subgoal_end_effector_base.load()

        self.door_handle_vis = VisualMarker(rgba_color=[1, 0, 0, 0.0], radius=self.door_handle_dist_thresh)
        self.door_handle_vis.load()

        # TODO: move robot joint id and name mapping to robot_locomotors.py
        self.id_to_name = {
            0: {"name": "ground", "links": {-1: "base", 0: "ground"}},
            1: {"name": "ground", "links": {-1: "base", 0: "ground"}},
            2: {"name": "ground", "links": {-1: "base", 0: "ground"}},
            3: {"name": "ground", "links": {-1: "base", 0: "ground"}},
        }
        self.id_to_name[self.door.body_id] = {"name": "door",
                                              "links": {-1: "world", 0: "base", 1: "door_leaf", 2: "door_knob"}}
        for i, wall in enumerate(self.walls):
            self.id_to_name[wall.body_id] = {"name": "wall%d" % (i + 1), "links": {-1: "world", 0: "wall"}}
        self.id_to_name[self.robots[0].robot_ids[0]] = {"name": "robot", "links": {
            -1: "base",
            0: "base_chassis",
            1: "jr2_fixed_body (wrapper)",
            2: "left wheel",
            3: "right wheel",
            4: "front_caster_pivot",
            5: "front_caster_wheel",
            6: "rear_caster_pivot",
            7: "rear_caster_wheel",
            8: "ext_imu_frame",
            9: "rear_laser",
            10: "front_laser",
            11: "lower_velodyne_frame",
            12: "occam",
            13: "occam_omni_optical",
            14: "upper_velodyne_frame",
            15: "scan",
            16: "gps_frame",
            17: "pan",
            18: "tilt",
            19: "camera",
            20: "camera_rgb_frame",
            21: "camera_rgb_optical_frame",
            22: "camera_depth_frame",
            23: "camera_depth_optical_frame",
            24: "eyes",
            25: "right_arm_attach",
            26: "m1n6s200_link_base",
            27: "m1n6s200_link_1 (shoulder)",
            28: "m1n6s200_link_2 (arm)",
            29: "m1n6s200_link_3 (elbow)",
            30: "m1n6s200_link_4 (forearm)",
            31: "m1n6s200_link_5 (wrist)",
            32: "m1n6s200_link_6 (hand)",
            33: "end_effector",
        }}
        self.num_object_classes = 4

    def set_subgoal(self, ideal_next_state):
        obs_avg = (self.observation_normalizer['sensor'][1] + self.observation_normalizer['sensor'][0]) / 2.0
        obs_mag = (self.observation_normalizer['sensor'][1] - self.observation_normalizer['sensor'][0]) / 2.0
        ideal_next_state = (ideal_next_state * obs_mag) + obs_avg
        self.subgoal_end_effector.set_position(ideal_next_state)
        self.subgoal_end_effector_base.set_position([ideal_next_state[0], ideal_next_state[1], 0])

    def set_subgoal_type(self, only_base=True):
        if only_base:
            # Make the marker for the end effector completely transparent
            self.subgoal_end_effector.set_color([0, 0, 0, 0.0])
        else:
            self.subgoal_end_effector.set_color([0, 0, 0, 0.8])

    def reset_interactive_objects(self):
        # close the door
        p.resetJointState(self.door.body_id, self.door_axis_link_id, targetValue=0.0, targetVelocity=0.0)
        if self.cid is not None:
            p.removeConstraint(self.cid)
            self.cid = None

    def reset_initial_and_target_pos(self):
        if self.arena == "only_ll" or self.arena == "only_ll_obstacles":
            if self.random_position:
                pos = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0]
            else:
                pos = [0.0, 0.0, 0.0]
        elif self.arena == "simple_hl_ll":
            if self.random_position:
                pos = [np.random.uniform(1, 2), np.random.uniform(-2, 2), 0]
            else:
                pos = [1.0, 0.0, 0.0]
        elif self.arena == "complex_hl_ll":
            if self.random_position:
                pos = [np.random.uniform(-2, -1.7), np.random.uniform(4.5, 5), 0]
            else:
                pos = [-2, 4, 0.0]

        # self.robots[0].set_position(pos=[pos[0], pos[1], pos[2] + 0.1])
        self.robots[0].set_position(pos=[pos[0], pos[1], pos[2]])

        self.initial_pos = pos

        if self.arena == "only_ll" or self.arena == "only_ll_obstacles":
            self.robots[0].set_orientation(
                orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz')
            )
        elif self.arena == "simple_hl_ll":
            if self.random_position:
                self.robots[0].set_orientation(
                    orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz')
                )
            else:
                self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(0, 0, np.pi), 'wxyz'))
        elif self.arena == "complex_hl_ll":

            self.robots[0].set_orientation(
                orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz')
            )
            # if self.random_position:
            #     self.robots[0].set_orientation(
            #         orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz')
            #     )
            # else:
            #     self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(0, 0, 0), 'wxyz'))

            self.initial_pos = pos

            if self.random_position:
                self.robots[0].set_orientation(
                    orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz')
                )
            else:
                self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(0, 0, 0), 'wxyz'))

        # # wait for the base to fall down to the ground and for the arm to move to its initial position
        # for _ in range(int(0.5 / self.physics_timestep)):
        #     self.simulator_step()


        if self.arena == "only_ll" or self.arena == "only_ll_obstacles":
            self.target_pos = [-100, -100, 0]
        elif self.arena == "simple_hl_ll" or self.arena == "complex_hl_ll":
            if self.random_position:
                self.target_pos = [np.random.uniform(-2, -1), np.random.uniform(-2, 2), 0.0]
            else:
                self.target_pos = np.array([-1.5, 0.0, 0.0])

        self.door_handle_vis.set_position(pos=np.array(p.getLinkState(self.door.body_id, self.door_handle_link_id)[0]))

        collision_links = []
        for _ in range(self.simulator_loop):
            self.simulator_step()
            collision_links.append(list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0])))
        collision_links = self.filter_collision_links(collision_links)
        no_collision = len(collision_links) == 0
        return no_collision

    def reset(self):
        self.reset_interactive_objects()
        self.stage = 0
        self.prev_stage = self.stage
        return super(InteractiveNavigateEnv, self).reset()

    # wrap the specified dimensions of the states to [-pi, pi]
    def wrap_to_pi(self, states, indices):
        states[indices] = states[indices] - 2.0 * np.pi * np.floor((states[indices] + np.pi) / (np.pi * 2))
        return states

    def get_state(self, collision_links=[]):
        state = super(InteractiveNavigateEnv, self).get_state(collision_links)
        # self.state_stats['sensor'].append(state['sensor'])
        # self.state_stats['auxiliary_sensor'].append(state['auxiliary_sensor'])
        if self.normalize_observation:
            for key in state:
                obs_min = self.observation_normalizer[key][0]
                obs_max = self.observation_normalizer[key][1]
                obs_avg = (self.observation_normalizer[key][1] + self.observation_normalizer[key][0]) / 2.0
                obs_mag = (self.observation_normalizer[key][1] - self.observation_normalizer[key][0]) / 2.0
                # clipped = np.clip(state[key], obs_min, obs_max)
                # if np.sum(state[key] == clipped) / float(state[key].shape[0]) < 0.8:
                #     print("WARNING: more than 20% of the observations are clipped for key: {}".format(key))
                state[key] = (np.clip(state[key], obs_min, obs_max) - obs_avg) / obs_mag  # normalize to [-1, 1]
        # self.state_stats['rgb'].append(state['rgb'])
        # self.state_stats['depth'].append(state['depth'])
        return state

    def get_additional_states(self):
        additional_states = self.robots[0].get_end_effector_position()
        assert len(additional_states) == self.additional_states_dim, 'additional states dimension mismatch'
        return additional_states

    def get_auxiliary_sensor(self, collision_links=[]):
        auxiliary_sensor = np.zeros(self.auxiliary_sensor_dim)
        robot_state = self.robots[0].calc_state()
        assert self.auxiliary_sensor_dim == 66
        assert robot_state.shape[0] == 31

        robot_state = self.wrap_to_pi(robot_state, np.arange(7, 28, 3))  # wrap wheel and arm joint pos to [-pi, pi]

        end_effector_pos = self.robots[0].get_end_effector_position() - self.robots[0].get_position()
        end_effector_pos = rotate_vector_3d(end_effector_pos, *self.robots[0].get_rpy())
        auxiliary_sensor[:3] = self.robots[0].get_position()  # x, y, z
        auxiliary_sensor[3:6] = end_effector_pos  # arm_x, arm_y_ arm_z (local)
        auxiliary_sensor[6:11] = robot_state[1:6]  # vx, vy, vz, roll, pitch
        auxiliary_sensor[11:46:5] = robot_state[7:28:3]  # pos for wheel 1, 2, arm joint 1, 2, 3, 4, 5
        auxiliary_sensor[12:47:5] = robot_state[8:29:3]  # vel for wheel 1, 2, arm joint 1, 2, 3, 4, 5
        auxiliary_sensor[13:48:5] = robot_state[9:30:3]  # trq for wheel 1, 2, arm joint 1, 2, 3, 4, 5
        auxiliary_sensor[14:49:5] = np.cos(robot_state[7:28:3])  # cos(pos) for wheel 1, 2, arm joint 1, 2, 3, 4, 5
        auxiliary_sensor[15:50:5] = np.sin(robot_state[7:28:3])  # sin(pos) for wheel 1, 2, arm joint 1, 2, 3, 4, 5
        auxiliary_sensor[46:49] = robot_state[28:31]  # v_roll, v_pitch, v_yaw

        roll, pitch, yaw = self.robots[0].get_rpy()
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        door_angle = p.getJointState(self.door.body_id, self.door_axis_link_id)[0]
        cos_door_angle, sin_door_angle = np.cos(door_angle), np.sin(door_angle)
        has_door_handle_in_hand = 1.0 if self.stage == self.stage_open_door else -1.0
        door_pos = np.array([0, 0, -0.02])
        target_pos = self.target_pos
        robot_pos = self.robots[0].get_position()
        door_pos_local = rotate_vector_3d(door_pos - robot_pos, roll, pitch, yaw)
        target_pos_local = rotate_vector_3d(target_pos - robot_pos, roll, pitch, yaw)
        collision_links_flatten = [item for sublist in collision_links for item in sublist]
        has_collision = 1.0 if len(collision_links_flatten) > 0 else -1.0

        auxiliary_sensor[49:52] = np.array([yaw, cos_yaw, sin_yaw])
        auxiliary_sensor[52:56] = np.array([door_angle, cos_door_angle, sin_door_angle, has_door_handle_in_hand])
        auxiliary_sensor[56:59] = target_pos
        auxiliary_sensor[59:62] = door_pos_local
        auxiliary_sensor[62:65] = target_pos_local
        auxiliary_sensor[65] = has_collision

        return auxiliary_sensor


    def filter_collision_links(self, collision_links):
        collision_links = super(InteractiveNavigateEnv, self).filter_collision_links(collision_links)

        # ignore collision between hand and door
        collision_links = [[item for item in sublist
                            if not (item[2] == self.door.body_id and item[3] in [32, 33])]
                           for sublist in collision_links]
        return collision_links

    def step(self, action):
        self.current_step += 1

        dist = np.linalg.norm(
            np.array(p.getLinkState(self.door.body_id, self.door_handle_link_id)[0]) -
            np.array(p.getLinkState(self.robots[0].robot_ids[0], self.jr_end_effector_link_id)[0])
        )
        # print('dist', dist)

        self.prev_stage = self.stage
        if self.stage == self.stage_get_to_door_handle and dist < self.door_handle_dist_thresh:
            assert self.cid is None
            self.cid = p.createConstraint(self.robots[0].robot_ids[0], self.jr_end_effector_link_id,
                                          self.door.body_id, self.door_handle_link_id,
                                          p.JOINT_POINT2POINT, [0, 0, 0],
                                          [0, 0.0, 0], [0, 0, 0])
            p.changeConstraint(self.cid, maxForce=500)
            self.stage = self.stage_open_door
            print("stage open_door")

        if self.stage == self.stage_open_door and p.getJointState(self.door.body_id, 1)[0] > self.door_angle:
            assert self.cid is not None
            p.removeConstraint(self.cid)
            self.cid = None
            self.stage = self.stage_get_to_target_pos
            print("stage get to target pos")

        door_angle = p.getJointState(self.door.body_id, self.door_axis_link_id)[0]

        # door is pushed in the wrong direction, gradually reset it back to the neutral state
        if door_angle < -0.01:
            max_force = 10000
            p.setJointMotorControl2(bodyUniqueId=self.door.body_id,
                                    jointIndex=self.door_axis_link_id,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0.0,
                                    positionGain=1,
                                    force=max_force)
        # door is pushed in the correct direction
        else:
            # if the door has not been opened, overwrite the previous position control with a trivial one
            if self.stage != self.stage_get_to_target_pos:
                max_force = 0
                p.setJointMotorControl2(bodyUniqueId=self.door.body_id,
                                        jointIndex=self.door_axis_link_id,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=door_angle,
                                        positionGain=0,
                                        velocityGain=0,
                                        force=max_force)

            # if the door has already been opened, try to set velocity to 0 so that it's more difficult for
            # the agent to move the door on its way to the target position
            else:
                max_force = 100
                p.setJointMotorControl2(bodyUniqueId=self.door.body_id,
                                        jointIndex=self.door_axis_link_id,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=door_angle,
                                        targetVelocity=0.0,
                                        positionGain=0,
                                        velocityGain=1,
                                        force=max_force)

        return super(InteractiveNavigateEnv, self).step(action)

    def get_potential(self):
        door_angle = p.getJointState(self.door.body_id, self.door_axis_link_id)[0]
        door_handle_pos = p.getLinkState(self.door.body_id, self.door_handle_link_id)[0]
        if self.stage == self.stage_get_to_door_handle:
            potential = l2_distance(door_handle_pos, self.robots[0].get_end_effector_position())
        elif self.stage == self.stage_open_door:
            potential = -door_angle
        elif self.stage == self.stage_get_to_target_pos:
            potential = l2_distance(self.target_pos, self.get_position_of_interest())
        return potential

    def get_reward(self, collision_links=[], action=None, info={}):
        reward = 0.0

        if self.reward_type == 'dense':
            if self.stage != self.prev_stage:
                # advance to the next stage
                self.potential = self.get_potential()
                # reward += self.success_reward / 2.0
            else:
                new_potential = self.get_potential()
                potential_reward = self.potential - new_potential
                reward += potential_reward * self.potential_reward_weight  # |potential_reward| ~= 0.1 per step
                self.potential = new_potential
        elif self.reward_type == "l2":
            new_potential = self.get_l2_potential()
            potential_reward = self.potential - new_potential
            reward += potential_reward * self.potential_reward_weight
            self.potential = new_potential
        elif self.reward_type == 'stage_sparse':
            if self.stage != self.prev_stage:
                reward += self.success_reward / 2.0

        base_moving = np.any(np.abs(action[:2]) >= 0.01)
        arm_moving = np.any(np.abs(action[2:]) >= 0.01)
        electricity_reward = float(base_moving) + float(arm_moving)
        self.energy_cost += electricity_reward
        reward += electricity_reward * self.electricity_reward_weight

        collision_links_flatten = [item for sublist in collision_links for item in sublist]
        collision_reward = float(len(collision_links_flatten) > 0)
        self.collision_step += int(len(collision_links_flatten) > 0)
        reward += collision_reward * self.collision_reward_weight  # |collision_reward| ~= 1.0 per step if collision
        info['collision_reward'] = collision_reward * self.collision_reward_weight  # expose collision reward to info
        # self.reward_stats.append(np.abs(collision_reward * self.collision_reward_weight))

        # goal reached
        if l2_distance(self.target_pos, self.get_position_of_interest()) < self.dist_tol:
            reward += self.success_reward  # |success_reward| = 10.0

        # death penalty

        floor_height = 0.0 if self.floor_num is None else self.scene.get_floor_height(self.floor_num)
        if self.robots[0].get_position()[2] > floor_height + self.death_z_thresh:
            reward -= self.success_reward * 1.0

        # push door the wrong way
        # door_angle = p.getJointState(self.door.body_id, self.door_axis_link_id)[0]
        # if door_angle > (10.0 / 180.0 * np.pi):
        #     reward -= self.success_reward * 1.0

        # print("get_reward (stage %d): %f" % (self.stage, reward))
        return reward, info

class NavigateRandomObstaclesEnv(NavigateEnv):
    def __init__(self,
                 config_file,
                 mode='headless',
                 action_timestep=1 / 10.0,
                 physics_timestep=1 / 240.0,
                 automatic_reset=False,
                 random_height=False,
                 device_idx=0,
    ):
        super(NavigateRandomObstaclesEnv, self).__init__(config_file,
                                                         mode=mode,
                                                         action_timestep=action_timestep,
                                                         physics_timestep=physics_timestep,
                                                         automatic_reset=automatic_reset,
                                                         device_idx=device_idx)
        self.random_height = random_height
        
        # wall = [pos, dim]
        self.walls = [[[0, 5, 0.501], [5, 0.2, 0.5]],
                      [[0, -5, 0.501], [5, 0.1, 0.5]],
                      [[5, 0, 0.501], [0.1, 5, 0.5]],
                      [[-5, 0, 0.501], [0.1, 5, 0.5]]]
        
        for i in range(len(self.walls)):
            wall = self.walls[i]
            box = BoxShape(pos=[wall[0][0], wall[0][1], wall[0][2]], dim=[wall[1][0], wall[1][1], wall[1][2]])            
            self.obstacle_ids.append(self.simulator.import_object(box))
        
        # Fix number of boxes and their positional range for now.
        self.obstacles_low_x, self.obstacles_high_x = -3.0, 3.0
        self.obstacles_low_y, self.obstacles_high_y = -3.0, 3.0
        self.num_obstacles = 7
        self.box_x, self.box_y, self.box_z = 0.2, 0.3, 0.3
        initial_box_pose = [0, 0, 0]
        self.boxes = []
        self.box_poses = []

        for _ in range(self.num_obstacles):
            box = BoxShape(pos=initial_box_pose, dim=[self.box_x, self.box_y, self.box_z], rgba_color=[1.0, 0.0, 0.0, 1.0])
            self.obstacle_ids.append(self.simulator.import_interactive_object(box))
            self.boxes.append(box)
            self.box_poses.append(initial_box_pose)

    def reset_obstacles(self):
        for i in range(self.num_obstacles):
            _, pos = self.scene.get_random_point(min_xy=-3.0, max_xy=3.0)
            self.boxes[i].set_position(pos=pos)
            self.box_poses[i] = pos


    def reset_initial_and_target_pos(self):
        self.reset_obstacles()
        # This will make the randomized initial position converge very quick.
        # floor, pos = self.scene.get_random_point(min_xy=self.initial_pos[0], max_xy=self.initial_pos[1])
        floor, pos = self.scene.get_random_point(min_xy=-2.0, max_xy=2.0)
        self.robots[0].set_position(pos=[pos[0], pos[1], pos[2] + 0.1])
        self.robots[0].set_orientation(
            orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz'))
        self.initial_pos = pos

        dist = 0.0
        overlap_obstacles = True # Whether target overlaps with obstacles.
        while dist < 1.0 or overlap_obstacles:
            _, self.current_target_position = self.scene.get_random_point_floor(floor, min_xy=self.target_pos[0], max_xy=self.target_pos[1], random_height=self.random_height)
            dist = l2_distance(self.initial_pos, self.current_target_position)
            overlap_obstacles = False
            for pos in self.box_poses:
                if l2_distance(self.current_target_position, pos) < self.box_y: # roughly set
                    overlap_obstacles = True
        
        # Check whether the agent collides with objects in the environment.
        collision_links = []
        for _ in range(self.simulator_loop):
            self.simulator_step()
            collision_links.append(list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0])))
        collision_links = self.filter_collision_links(collision_links)
        no_collision = len(collision_links) == 0
        return no_collision

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot',
                        '-r',
                        choices=['turtlebot', 'jr'],
                        required=True,
                        help='which robot [turtlebot|jr]')
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    parser.add_argument('--env_type',
                        choices=['deterministic', 'random', 'fixed_obstacles', 'random_obstacles', 'pedestrians', 'interactive'],
                        default='deterministic',
                        help='which environment type (deterministic | random |  fixed_obstacles random_obstacles | pedestrians | interactive')
   #  parser.add_argument('--layout',
                        # '-l',
                        # default=None,
                        # help='layout config file')

    args = parser.parse_args()

    if args.robot == 'turtlebot':
        config_filename = os.path.join(os.path.dirname(gibson2.__file__),
                                       '../examples/configs/turtlebot_p2p_nav_discrete.yaml') \
            if args.config is None else args.config
    elif args.robot == 'jr':
        config_filename = os.path.join(os.path.dirname(gibson2.__file__),
                                       '../examples/configs/jr2_reaching.yaml') \
            if args.config is None else args.config
    if args.env_type == 'deterministic':
        nav_env = NavigateEnv(config_file=config_filename,
                              mode=args.mode,
                              action_timestep=1.0 / 10.0,
                              physics_timestep=1.0 / 40.0)
    elif args.env_type == 'random':
        nav_env = NavigateRandomEnv(config_file=config_filename,
                                    mode=args.mode,
                                    action_timestep=1.0 / 10.0,
                                    physics_timestep=1.0 / 40.0)

    elif args.env_type == 'fixed_obstacles':
        nav_env = NavigateObstaclesEnv(config_file=config_filename,
                                    mode=args.mode,
                                    action_timestep=1.0 / 10.0,
                                    physics_timestep=1.0 / 40.0)
    elif args.env_type == 'random_obstacles':
        nav_env = NavigateRandomObstaclesEnv(config_file=config_filename,
                                    mode=args.mode,
                                    action_timestep=1.0 / 10.0,
                                    physics_timestep=1.0 / 40.0)
    elif args.env_type == 'pedestrians':
        nav_env = NavigatePedestriansEnv(config_file=config_filename,
                                    mode=args.mode,
                                    # layout=args.layout,
                                    action_timestep=1.0 / 20.0,
                                    physics_timestep=1.0 / 40.0)
    elif args.env_type == 'random_obstacles':
        nav_env = NavigateRandomObstaclesEnv(config_file=config_filename,
                                    mode=args.mode,
                                    action_timestep=1.0 / 10.0,
                                    physics_timestep=1.0 / 40.0)

    elif args.env_type == 'ig':
        nav_env = InteractiveGibsonNavigateEnv(config_file=config_filename,
                                               mode=args.mode,
                                               action_timestep=1.0 / 10.0,
                                               physics_timestep=1.0 / 40.0)
    else:
        nav_env = InteractiveNavigateEnv(config_file=config_filename,
                                         mode=args.mode,
                                         action_timestep=1.0 / 10.0,
                                         random_position=False,
                                         physics_timestep=1.0 / 40.0,
                                         arena='only_ll_obstacles')

    
    # # Sample code: manually set action using slide bar UI
    # debug_params = [
    #     p.addUserDebugParameter('link1', -1.0, 1.0, -0.5),
    #     p.addUserDebugParameter('link2', -1.0, 1.0, 0.5),
    #     p.addUserDebugParameter('link3', -1.0, 1.0, 0.5),
    #     p.addUserDebugParameter('link4', -1.0, 1.0, 0.5),
    #     p.addUserDebugParameter('link5', -1.0, 1.0, 0.0),
    # ]

    for episode in range(10000):
        print('Episode: {}'.format(episode))
        start = time.time()
        nav_env.reset()
        for i in range(nav_env.config.get('max_step', 500)):  # 500 steps, 50s world time
            action = nav_env.action_space.sample()
            # action[:] = 0

            # if nav_env.stage == 0:
            #     action[:2] = 0.5
            # elif nav_env.stage == 1:
            #     action[:2] = -0.5

            # action = np.zeros(nav_env.action_space.shape)
            # debug_param_values = [p.readUserDebugParameter(debug_param) for debug_param in debug_params]
            # action[2:] = np.array(debug_param_values)
            #state, reward, done, info = nav_env.step([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
            state, reward, done, info = nav_env.step(action)
            #state, reward, done, _ = nav_env.step([-0.9, 0.0])            
            # print(reward)

            if done:
                print('Episode finished after {} timesteps'.format(i + 1))
                break
        # print(time.time() - start)
    nav_env.clean()
