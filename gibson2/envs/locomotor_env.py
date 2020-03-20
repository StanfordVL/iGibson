from gibson2.core.physics.interactive_objects import VisualMarker, InteractiveObj, BoxShape
#from semantic_segmentation_pytorch.models import ModelBuilder
#from semantic_segmentation_pytorch.utils import colorEncode
#import semantic_segmentation_pytorch
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
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image
import string
import random


Episode = collections.namedtuple('Episode',
                                 [
                                     # 'env',
                                     # 'agent',
                                     # 'initial_pos',
                                     # 'target_pos',
                                     # 'geodesic_distance',
                                     # 'shortest_path',
                                     # 'agent_trajectory',
                                     # 'object_files',
                                     # 'object_trajectory',
                                     'success',
                                     # 'path_efficiency',
                                     # 'kinematic_disturbance',
                                     # 'dynamic_disturbance_a',
                                     # 'dynamic_disturbance_b',
                                     # 'collision_step',
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
        # ignore the agent's collision with these body ids, typically ids of the ground
        self.collision_ignore_body_b_ids = set(self.config.get('collision_ignore_body_b_ids', []))
        self.collision_ignore_link_a_ids = set(self.config.get('collision_ignore_link_a_ids', []))

        # discount factor
        self.discount_factor = self.config.get('discount_factor', 1.0)

        # output: sensors
        self.output = self.config['output']
        self.n_horizontal_rays = self.config.get('n_horizontal_rays', 128)
        self.n_vertical_beams = self.config.get('n_vertical_beams', 9)
        self.scan_noise_rate = self.config.get('scan_noise_rate', 0.0)
        self.depth_noise_rate = self.config.get('depth_noise_rate', 0.0)

        # TODO: sensor: observations that are passed as network input, e.g. target position in local frame
        # TODO: auxiliary sensor: observations that are not passed as network input, but used to maintain the same
        # subgoals for the next T time steps, e.g. agent pose in global frame
        self.sensor_dim = self.additional_states_dim
        self.action_dim = self.robots[0].action_dim

        observation_space = OrderedDict()
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
            # self.depth_space = gym.spaces.Box(low=-np.inf,
            #                                   high=np.inf,
            #                                   shape=(self.config.get('resolution', 64),
            #                                          self.config.get('resolution', 64),
            #                                          1),
            #                                   dtype=np.float32)
            self.depth_space = gym.spaces.Box(low=0.0,
                                              high=1.0,
                                              shape=(self.config.get('resolution', 64),
                                                     self.config.get('resolution', 64),
                                                     1),
                                              dtype=np.float32)
            observation_space['depth'] = self.depth_space
        if 'rgbd' in self.output:
            self.rgbd_space = gym.spaces.Box(low=0.0,
                                             high=1.0,
                                             shape=(self.config.get('resolution', 64),
                                                    self.config.get('resolution', 64),
                                                    4),
                                             dtype=np.float32)
            observation_space['rgbd'] = self.rgbd_space
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
            # self.scan_space = gym.spaces.Box(low=-np.inf,
            #                                  high=np.inf,
            #                                  shape=(self.n_horizontal_rays * self.n_vertical_beams, 3),
            #                                  dtype=np.float32)
            self.scan_space = gym.spaces.Box(low=0.0,
                                             high=1.0,
                                             shape=(self.n_horizontal_rays * self.n_vertical_beams, 1),
                                             dtype=np.float32)
            observation_space['scan'] = self.scan_space
        if 'rgb_filled' in self.output:  # use filler
            self.comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
            self.comp = torch.nn.DataParallel(self.comp).cuda()
            self.comp.load_state_dict(
                torch.load(os.path.join(gibson2.assets_path, 'networks', 'model.pth')))
            self.comp.eval()
        # if 'seg_pred' in self.output:
        #     # torch.cuda.set_device(1)
        #     encoder_weights = os.path.join(os.path.dirname(semantic_segmentation_pytorch.__file__),
        #                                    # 'ckpt/ade20k-resnet18dilated-ppm_deepsup/encoder_epoch_20.pth')
        #                                    'ckpt/ade20k-mobilenetv2dilated-c1_deepsup/encoder_epoch_20.pth')
        #
        #     self.seg_encoder = ModelBuilder.build_encoder(
        #         arch='mobilenetv2dilated',
        #         # arch='resnet18dilated',
        #         weights=encoder_weights)
        #     self.seg_encoder.cuda()
        #     self.seg_encoder.eval()
        #
        #     decoder_weights = os.path.join(os.path.dirname(semantic_segmentation_pytorch.__file__),
        #                                    # 'ckpt/ade20k-resnet18dilated-ppm_deepsup/decoder_epoch_20.pth')
        #                                    'ckpt/ade20k-mobilenetv2dilated-c1_deepsup/decoder_epoch_20.pth')
        #     self.seg_decoder = ModelBuilder.build_decoder(
        #         # arch='ppm_deepsup',
        #         # fc_dim=512,
        #         arch='c1_deepsup',
        #         fc_dim=320,
        #         num_class=150,
        #         weights=decoder_weights,
        #         use_softmax=True)
        #     self.seg_decoder.cuda()
        #     self.seg_decoder.eval()
        #     color_path = os.path.join(os.path.dirname(semantic_segmentation_pytorch.__file__), 'data/color150.mat')
        #     self.seg_colors = loadmat(color_path)['colors']
        #
        #     self.seg_normalizer = transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225])
        #
        #     self.seg_pred_space = gym.spaces.Box(low=0.0,
        #                                          high=1.0,
        #                                          shape=(self.config.get('resolution', 64) // 8,
        #                                                 self.config.get('resolution', 64) // 8,
        #                                                 320),
        #                                          dtype=np.float32)
        #     observation_space['seg_pred'] = self.seg_pred_space

        self.observation_space = gym.spaces.Dict(observation_space)
        self.action_space = self.robots[0].action_space

        # add visual objects
        self.visual_object_at_initial_target_pos = self.config.get('visual_object_at_initial_target_pos', False)

        if self.visual_object_at_initial_target_pos:
            cyl_length = 0.2
            self.initial_pos_vis_obj = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                                    rgba_color=[1, 0, 0, 0.3],
                                                    radius=self.dist_tol,
                                                    length=cyl_length,
                                                    initial_offset=[0, 0, cyl_length / 2.0])
            self.target_pos_vis_obj = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                                   rgba_color=[0, 0, 1, 0.3],
                                                   radius=self.dist_tol,
                                                   length=cyl_length,
                                                   initial_offset=[0, 0, cyl_length / 2.0])
            self.initial_pos_vis_obj.load()
            if self.config.get('target_visual_object_visible_to_agent', False):
                self.simulator.import_object(self.target_pos_vis_obj, class_id=255)
            else:
                self.target_pos_vis_obj.load()

    def get_additional_states(self):
        relative_position = self.target_pos - self.robots[0].get_position()
        # rotate relative position back to body point of view
        additional_states = rotate_vector_3d(relative_position, *self.robots[0].get_rpy())

        if self.config['task'] == 'reaching':
            end_effector_pos = self.robots[0].get_end_effector_position() - self.robots[0].get_position()
            end_effector_pos = rotate_vector_3d(end_effector_pos, *self.robots[0].get_rpy())
            additional_states = np.concatenate((additional_states, end_effector_pos))
        assert len(additional_states) == self.additional_states_dim, 'additional states dimension mismatch'

        return additional_states

    def get_auxiliary_sensor(self, collision_links=[]):
        return np.array([])

    def add_naive_noise_to_sensor(self, sensor_reading, noise_rate, noise_value=1.0):
        if noise_rate <= 0.0:
            return sensor_reading

        assert len(sensor_reading[(sensor_reading < 0.0) | (sensor_reading > 1.0)]) == 0,\
            'sensor reading has to be between [0.0, 1.0]'

        valid_mask = np.random.choice(2, sensor_reading.shape, p=[noise_rate, 1.0 - noise_rate])
        # set invalid values to be the maximum range (e.g. depth and scan)
        sensor_reading[valid_mask == 0] = noise_value
        return sensor_reading

    def get_depth(self):
        depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
        if self.config['robot'] == 'Turtlebot':
            # ASUS Xtion PRO LIVE
            low = 0.8
            high = 3.5
        elif self.config['robot'] == 'Fetch':
            # Primesense Carmine 1.09 short-range RGBD sensor
            low = 0.35
            high = 3.0  # http://xtionprolive.com/primesense-carmine-1.09
            # high = 1.4  # https://www.i3du.gr/pdf/primesense.pdf
        elif self.config['robot'] == 'Locobot':
            # https://store.intelrealsense.com/buy-intel-realsense-depth-camera-d435.html
            low = 0.1
            high = 10.0
        else:
            assert False, 'unknown robot for RGBD observation'

        # 0.0 is a special value for invalid entries
        depth[depth < low] = 0.0
        depth[depth > high] = 0.0

        # re-scale depth to [0.0, 1.0]
        depth /= high
        depth = self.add_naive_noise_to_sensor(depth, self.depth_noise_rate, noise_value=0.0)

        return depth

    def get_rgb(self):
        return self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]

    def get_pc(self):
        return self.simulator.renderer.render_robot_cameras(modes=('3d'))[0]

    def get_normal(self):
        return self.simulator.renderer.render_robot_cameras(modes='normal')

    def get_seg(self):
        seg = self.simulator.renderer.render_robot_cameras(modes='seg')[0][:, :, 0:1]
        if self.num_object_classes is not None:
            seg = np.clip(seg * 255.0 / self.num_object_classes, 0.0, 1.0)
        return seg

    def get_seg_pred(self):
        rgb = self.get_rgb()
        with torch.no_grad():
            width = rgb.shape[0]
            height = int(width * (480.0 / 640.0))
            half_diff = int((width - height) / 2)
            rgb_cropped = rgb[half_diff:half_diff + height, :]
            rgb_cropped = (rgb_cropped * 255).astype(np.uint8)
            tensor = transforms.ToTensor()(rgb_cropped)
            img = self.seg_normalizer(tensor).unsqueeze(0).cuda()
            seg_pred = self.seg_encoder(img)
            seg_pred = seg_pred[0][0].permute(1, 2, 0).cpu().numpy()
            return seg_pred

            # # visualize predicted segmentation mask
            # scores = self.seg_decoder(self.seg_encoder(img, return_feature_maps=True), segSize=(height, width))
            # _, pred = torch.max(scores, dim=1)
            # pred = pred.squeeze(0).cpu().numpy().astype(np.int32)
            # pred_color = colorEncode(pred, self.seg_colors).astype(np.uint8)
            # pred_color = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
            #
            # depth_cropped = depth[half_diff:half_diff + height, :]
            # low = 0.8
            # high = 3.5
            # invalid = depth_cropped == 0.0
            # depth_cropped[depth_cropped < low] = low
            # depth_cropped[depth_cropped > high] = high
            # depth_cropped[invalid] = 0.0
            # depth_cropped /= high
            # depth_cropped = (depth_cropped * 255).astype(np.uint8)
            # depth_cropped = np.tile(depth_cropped, (1, 1, 3))
            #
            # rgb_cropped = cv2.cvtColor(rgb_cropped, cv2.COLOR_RGB2BGR)
            #
            # vis = np.concatenate((rgb_cropped, depth_cropped, pred_color), axis=1)
            # cv2.imshow('vis', vis)

    def get_scan(self):
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
            min_laser_dist = 0.0
            laser_link_name = 'laser_link'
        else:
            assert False, 'unknown robot for LiDAR observation'

        assert self.n_vertical_beams == 1, 'scan can only handle one vertical beam for now.'

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
        hit_fraction = self.add_naive_noise_to_sensor(hit_fraction, self.scan_noise_rate)

        scan = np.expand_dims(hit_fraction, 1)
        return scan

        # assert 'scan_link' in self.robots[0].parts, "Requested scan but no scan_link"
        # pose_camera = self.robots[0].parts['scan_link'].get_pose()
        # angle = np.arange(0, 2 * np.pi, 2 * np.pi / float(self.n_horizontal_rays))
        # elev_bottom_angle = -30. * np.pi / 180.
        # elev_top_angle = 10. * np.pi / 180.
        # elev_angle = np.arange(elev_bottom_angle, elev_top_angle,
        #                        (elev_top_angle - elev_bottom_angle) / float(self.n_vertical_beams))
        # orig_offset = np.vstack([
        #     np.vstack([np.cos(angle),
        #                np.sin(angle),
        #                np.repeat(np.tan(elev_ang), angle.shape)]).T for elev_ang in elev_angle
        # ])
        # transform_matrix = quat2mat([pose_camera[-1], pose_camera[3], pose_camera[4], pose_camera[5]])
        # offset = orig_offset.dot(np.linalg.inv(transform_matrix))
        # pose_camera = pose_camera[None, :3].repeat(self.n_horizontal_rays * self.n_vertical_beams, axis=0)
        #
        # results = p.rayTestBatch(pose_camera, pose_camera + offset * 30)
        # hit = np.array([item[0] for item in results])
        # dist = np.array([item[2] for item in results])
        #
        # valid_pts = (dist < 1. - 1e-5) & (dist > 0.1 / 30) & (hit != self.robots[0].robot_ids[0]) & (hit != -1)
        # dist[~valid_pts] = 0.0  # zero out invalid pts
        # dist *= 30
        #
        # xyz = np.expand_dims(dist, 1) * orig_offset
        # state['scan'] = xyz

    def get_state(self, collision_links=[]):
        # calculate state
        sensor_state = self.get_additional_states()
        auxiliary_sensor = self.get_auxiliary_sensor(collision_links)

        # rgb = self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        #
        # depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
        # depth = np.clip(depth / 8.0, 0.0, 1.0)
        # depth = 1.0 - depth  # flip black/white
        # seg = self.simulator.renderer.render_robot_cameras(modes='seg')[0][:, :, 0:1]
        # if self.num_object_classes is not None:
        #     seg = np.clip(seg * 255.0 / self.num_object_classes, 0.0, 1.0)
        # cv2.imshow('rgb', rgb)
        # cv2.imwrite('test.jpg', (rgb * 255).astype(np.uint8))
        # cv2.imwrite('test_depth.jpg', (depth * 255).astype(np.uint8))

        # print(cv2.imwrite('button/%d_%d.jpg' % (self.current_episode, self.current_step), (rgb * 255).astype(np.uint8)))
        # print(cv2.imwrite('button/%d_%d_depth.jpg' % (self.current_episode, self.current_step), (depth * 255).astype(np.uint8)))
        # assert False
        # cv2.imshow('rgb', rgb)
        # cv2.imshow('depth', depth)

        # assert False

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
            state['rgb'] = self.get_rgb()
        if 'depth' in self.output:
            state['depth'] = self.get_depth()
        if 'pc' in self.output:
            state['pc'] = self.get_pc()
        if 'rgbd' in self.output:
            rgb = self.get_rgb()
            depth = self.get_depth()
            state['rgbd'] = np.concatenate((rgb, depth), axis=2)
        if 'normal' in self.output:
            state['normal'] = self.get_normal()
        if 'seg' in self.output:
            state['seg'] = self.get_seg()
        if 'depth_seg' in self.output:
            depth = self.get_depth()
            seg = self.get_seg()
            depth_seg = np.concatenate((depth, seg), axis=2)
            state['depth_seg'] = depth_seg
        if 'rgb_filled' in self.output:
            with torch.no_grad():
                tensor = transforms.ToTensor()((state['rgb'] * 255).astype(np.uint8)).cuda()
                rgb_filled = self.comp(tensor[None, :, :, :])[0].permute(1, 2, 0).cpu().numpy()
                state['rgb_filled'] = rgb_filled
        if 'seg_pred' in self.output:
            state['seg_pred'] = self.get_seg_pred()
        if 'pointgoal' in self.output:
            state['pointgoal'] = sensor_state[:2]
        if 'scan' in self.output:
            state['scan'] = self.get_scan()
        return state

    def run_simulation(self):
        collision_links = []
        for _ in range(self.simulator_loop):
            self.simulator_step()
            collision_links.append(list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0])))
        return self.filter_collision_links(collision_links)

    def filter_collision_links(self, collision_links):
        new_collision_links = []
        for collision_per_sim_step in collision_links:
            new_collision_per_sim_step = []
            for item in collision_per_sim_step:
                # ignore collision with body b
                if item[2] in self.collision_ignore_body_b_ids:
                    continue

                # ignore collision with robot link a
                if item[3] in self.collision_ignore_link_a_ids:
                    continue

                # ignore self collision with robot link a (body b is robot itself)
                if item[2] == self.robots[0].robot_ids[0] and item[4] in self.collision_ignore_link_a_ids:
                    continue

                new_collision_per_sim_step.append(item)
            new_collision_links.append(new_collision_per_sim_step)
        return new_collision_links

        # return [[item for item in collision_per_sim_step
        #          if item[2] not in self.collision_ignore_body_b_ids and
        #          item[3] not in self.collision_ignore_link_a_ids]
        #         for collision_per_sim_step in collision_links]

    def get_position_of_interest(self):
        if self.config['task'] == 'pointgoal':
            return self.robots[0].get_position()
        elif self.config['task'] == 'reaching':
            return self.robots[0].get_end_effector_position()

    def get_potential(self):
        return l2_distance(self.target_pos, self.get_position_of_interest())

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

        collision_reward = float(len(collision_links_flatten) > 0)
        self.collision_step += int(len(collision_links_flatten) > 0)
        info['collision_reward'] = collision_reward * self.collision_reward_weight  # expose collision reward to info
        reward += collision_reward * self.collision_reward_weight  # |collision_reward| ~= 1.0 per step if collision

        # goal reached
        if l2_distance(self.target_pos, self.get_position_of_interest()) < self.dist_tol:
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
        if l2_distance(self.target_pos, self.get_position_of_interest()) < self.dist_tol:
            print("GOAL")
            done = True
            info['success'] = True

        elif self.robots[0].get_position()[2] > floor_height + self.death_z_thresh:
            print("DEATH")
            done = True
            info['success'] = False

        elif self.current_step >= self.max_step:
            done = True
            info['success'] = False

        # elif door_angle < (-10.0 / 180.0 * np.pi):
        #     print("WRONG PUSH")

        # elif max_force > 500:
        #     print("TOO MUCH FORCE")
        #     done = True
        #     info['success'] = False

        if done:
        #     info['episode_length'] = self.current_step
        #     info['path_length'] = self.path_length
        #     info['collision_step'] = self.collision_step
        #     info['energy_cost'] = self.energy_cost
        #     info['stage'] = self.stage
        #
        #     shortest_path, geodesic_distance = self.scene.get_shortest_path(self.floor_num,
        #                                                                     self.initial_pos[:2],
        #                                                                     self.target_pos[:2],
        #                                                                     entire_path=True)
        #     floor_height = self.scene.get_floor_height(self.floor_num)
        #     shortest_path = np.array([np.array([path[0], path[1], floor_height]) for path in shortest_path])
        #     min_kin_dist = self.path_length * self.robots[0].robot_mass
        #     kinematic_disturbance = min_kin_dist / (min_kin_dist + self.kinematic_disturbance)
        #     min_dyn_dist = self.current_step * self.robots[0].robot_mass * 9.8
        #     dynamic_disturbance_a = min_dyn_dist / (min_dyn_dist + self.dynamic_disturbance_a)
        #     dynamic_disturbance_b = self.current_step / float(self.current_step + self.dynamic_disturbance_b)
        #     object_files = [obj.filename for obj in self.interactive_objects]
            episode = Episode(
                # env=self.scene.model_id,
                # agent=self.robots[0].model_file,
                # initial_pos=self.initial_pos,
                # target_pos=self.target_pos,
                # geodesic_distance=geodesic_distance,
                # shortest_path=shortest_path,
                # agent_trajectory=np.array(self.agent_trajectory),
                # object_files=object_files,
                # object_trajectory=np.array(self.object_trajectory),
                success=float(info['success']),
                # path_efficiency=min(1.0, geodesic_distance / self.path_length),
                # kinematic_disturbance=kinematic_disturbance,
                # dynamic_disturbance_a=dynamic_disturbance_a,
                # dynamic_disturbance_b=dynamic_disturbance_b,
                # collision_step=self.collision_step,
            )
            self.stored_episodes.append(episode)

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
        if action is not None:
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
            self.reset_initial_and_target_pos()
            valid, collision_links_flatten = self.test_valid_position()
            if valid:
                return
            # for collision_link in collision_links_flatten:
            #     print(collision_link)
            # print('reset agent failed')
            # embed()
        # print('Failed to reset robot without collision' + '-' * 30)
        # for collision_link in collision_links_flatten:
        #     print(collision_link)
        # random_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        # p.saveBullet(random_string + '.bullet')
        # p.saveWorld(random_string + '.py')
        # raise Exception("Failed to reset robot without collision")
        print("WARNING: Failed to reset robot without collision")

    def reset_initial_and_target_pos(self):
        self.robots[0].set_position(pos=self.initial_pos)
        self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(*self.initial_orn), 'wxyz'))

    def test_valid_position(self):
        # assume joint velocity control, set velocity to 0 when testing whether the initial configuration is valid
        assert self.robots[0].control == 'velocity'
        self.robots[0].apply_real_action(np.zeros(self.action_dim))
        collision_links = self.run_simulation()
        collision_links_flatten = [item for sublist in collision_links for item in sublist]
        return len(collision_links_flatten) == 0, collision_links_flatten

    def before_reset_agent(self):
        return

    def after_reset_agent(self):
        return

    def reset(self):
        """
        Reset the agent to a collision-free start point
        """

        self.current_episode += 1
        self.before_reset_agent()
        self.reset_agent()
        self.after_reset_agent()

        # set position for visual objects
        if self.visual_object_at_initial_target_pos:
            self.initial_pos_vis_obj.set_position(self.initial_pos)
            self.target_pos_vis_obj.set_position(self.target_pos)

        self.simulator.sync()
        state = self.get_state()

        if self.reward_type == 'l2':
            self.potential = self.get_l2_potential()
        elif self.reward_type == 'dense':
            self.potential = self.get_potential()

        self.current_step = 0
        self.collision_step = 0
        self.kinematic_disturbance = 0.0
        self.dynamic_disturbance_a = 0.0
        self.dynamic_disturbance_b = 0.0
        self.path_length = 0.0
        self.agent_trajectory = []
        self.object_trajectory = []
        self.interactive_objects_collided = set()
        self.energy_cost = 0.0

        return state


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

    def reset_initial_and_target_pos(self):
        floor, self.initial_pos = self.scene.get_random_point_floor(self.floor_num, self.random_height)

        max_trials = 100
        dist = 0.0
        # Need to change to 5 meter if we need to add obstacles
        for _ in range(max_trials):  # if initial and target positions are < 1 meter away from each other, reinitialize
            _, self.target_pos = self.scene.get_random_point_floor(self.floor_num, self.random_height)
            dist = l2_distance(self.initial_pos, self.target_pos)
            if dist > 1.0:
                break
        if dist < 1.0:
            raise Exception("Failed to find initial and target pos that are >1m apart")
        self.robots[0].set_position(pos=[self.initial_pos[0],
                                         self.initial_pos[1],
                                         self.initial_pos[2] + self.random_init_z_offset])
        self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz'))

    def reset(self):
        self.floor_num = self.scene.get_random_floor()
        self.scene.reset_floor(floor=self.floor_num, additional_elevation=0.05)
        state = super(NavigateRandomEnv, self).reset()
        return state


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


class InteractiveGibsonNavigateSim2RealEnv(NavigateRandomEnv):
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
        super(InteractiveGibsonNavigateSim2RealEnv, self).__init__(config_file,
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

        self.gt_pos = []
        self.ns_pos = []
        # self.linear_vel = []
        # self.angular_vel = []

        # self.eyes_vis = VisualMarker(rgba_color=[1, 0, 0, 1.0], radius=0.03)
        # self.eyes_vis.load()
        resolution = self.config.get('resolution', 64)
        width = resolution
        height = int(width * (480.0 / 640.0))
        if 'rgbd' in self.output:
            self.observation_space.spaces['rgbd'] = gym.spaces.Box(low=0.0,
                                                                   high=1.0,
                                                                   shape=(height, width, 4),
                                                                   dtype=np.float32)
        if 'seg_pred' in self.output:
            self.observation_space.spaces['seg_pred'] = gym.spaces.Box(low=-np.inf,
                                                                       high=np.inf,
                                                                       shape=(height // 8, width // 8, 320),
                                                                       dtype=np.float32)
        if 'rgb' in self.output:
            self.observation_space.spaces['rgb'] = gym.spaces.Box(low=0.0,
                                                                  high=1.0,
                                                                  shape=(height, width, 3),
                                                                  dtype=np.float32)
        if 'depth' in self.output:
            self.observation_space.spaces['depth'] = gym.spaces.Box(low=0.0,
                                                                    high=1.0,
                                                                    shape=(height, width, 1),
                                                                    dtype=np.float32)

        # self.scan_vis = VisualMarker(rgba_color=[1, 0, 0, 1.0], radius=0.05)
        # self.scan_vis.load()

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

        # cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('depth', 400, 400)
        # cv2.namedWindow('scan', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('scan', 400, 400)
        # cv2.namedWindow('collision', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('collision', 400, 400)

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

    def global_to_local(self, pos, cur_pos, cur_rot):
        return rotate_vector_3d(pos - cur_pos, *cur_rot)

    def get_additional_states(self):
        pos_noise = 0.0
        cur_pos = self.robots[0].get_position()
        cur_pos[:2] += np.random.normal(0, pos_noise, 2)
        # self.gt_pos.append(self.robots[0].get_position())
        # self.ns_pos.append(cur_pos)

        rot_noise = 0.0 / 180.0 * np.pi
        cur_rot = self.robots[0].get_rpy()
        cur_rot = (cur_rot[0], cur_rot[1], cur_rot[2] + np.random.normal(0, rot_noise))

        target_pos_local = self.global_to_local(self.target_pos, cur_pos, cur_rot)[:2]
        linear_velocity_local = rotate_vector_3d(self.robots[0].robot_body.velocity(), *cur_rot)[:2]
        angular_velocity_local = rotate_vector_3d(self.robots[0].robot_body.angular_velocity(), *cur_rot)[2:3]
        # print('linear', self.robots[0].robot_body.velocity())
        # print('linear_local', linear_velocity_local)
        # print('angular', self.robots[0].robot_body.angular_velocity())
        # print('angular_local', angular_velocity_local)

        # linear_vel = np.linalg.norm(self.robots[0].robot_body.velocity())
        # angular_vel = np.linalg.norm(self.robots[0].robot_body.angular_velocity())
        # self.linear_vel.append(linear_vel)
        # self.angular_vel.append(angular_vel)

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
        # self.eyes_vis.set_position(pos=self.robots[0].eyes.get_position())

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
                                            target_pos_local,
                                            linear_velocity_local,
                                            angular_velocity_local))
        # cache results for reward calculation
        additional_states = target_pos_local
        self.new_potential = geodesic_dist
        assert len(additional_states) == self.additional_states_dim, 'additional states dimension mismatch'
        return additional_states

    def crop_rect_image(self, img):
        width = img.shape[0]
        height = int(width * (480.0 / 640.0))
        half_diff = int((width - height) / 2)
        img = img[half_diff:half_diff + height, :]
        return img

    def get_state(self, collision_links=[]):
        state = super(InteractiveGibsonNavigateSim2RealEnv, self).get_state(collision_links)
        for modality in ['rgb', 'rgbd', 'depth']:
            if modality in state:
                state[modality] = self.crop_rect_image(state[modality])
        return state

    def get_potential(self):
        return self.new_potential

    def reset_additional_objects(self):
        for obj in self.additional_objects:
            while True:
                # _, pos = self.scene.get_random_point_floor(self.floor_num, self.random_height)
                pos_without_z = self.shortest_path[np.random.randint(self.shortest_path.shape[0])]
                pos = [
                    pos_without_z[0] + np.random.uniform(-0.2, 0.2),
                    pos_without_z[1] + np.random.uniform(-0.2, 0.2),
                    self.scene.get_floor_height(self.floor_num)
                ]
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
        # self.reset_additional_objects()
        self.new_potential = None
        # if len(self.linear_vel) > 0:
        #     print('linear', np.mean(self.linear_vel), np.max(self.linear_vel))
        #     print('angular', np.mean(self.angular_vel), np.max(self.angular_vel))
        # if len(self.gt_pos) > 0:
        #     self.gt_pos = np.array(self.gt_pos)
        #     self.ns_pos = np.array(self.ns_pos)
        #     plt.figure()
        #     plt.scatter(self.gt_pos[:, 0], self.gt_pos[:, 1], c='r')
        #     plt.scatter(self.ns_pos[:, 0], self.ns_pos[:, 1], c='b')
        #     plt.show()
        #     self.gt_pos = []
        #     self.ns_pos = []

        state = NavigateEnv.reset(self)
        return state

    def after_reset_agent(self):
        source = self.robots[0].get_position()[:2]
        target = self.target_pos[:2]
        shortest_path, _ = self.scene.get_shortest_path(self.floor_num, source, target, entire_path=True)
        self.shortest_path = shortest_path
        self.reset_additional_objects()
        # robot_z = self.robots[0].get_position()[2]
        # if self.visualize_waypoints and self.mode == 'gui':
        #     for i in range(1000):
        #         self.waypoints_vis[i].set_position(pos=np.array([0.0, 0.0, 0.0]))
        #     for i in range(min(1000, self.shortest_path.shape[0])):
        #         self.waypoints_vis[i].set_position(pos=np.array([self.shortest_path[i][0],
        #                                                          self.shortest_path[i][1],
        #                                                          robot_z]))

    def before_simulation(self):
        robot_position = self.robots[0].get_position()
        object_positions = [obj.get_position() for obj in self.interactive_objects]
        return robot_position, object_positions

    def after_simulation(self, cache, collision_links):
        robot_position, object_positions = cache

        # collision_links_flatten = [item for sublist in collision_links for item in sublist]
        # if len(collision_links_flatten) > 0:
        #     self.dynamic_disturbance_a += np.mean([
        #         np.linalg.norm(
        #             np.sum([elem[9] * np.array(elem[7]) for elem in sublist], axis=0)  # sum of all forces
        #         )
        #         for sublist in collision_links])
        #     collision_objects = set([col[2] for col in collision_links_flatten])
        #     self.dynamic_disturbance_b += len(collision_objects)
        #     self.interactive_objects_collided |= collision_objects
        #
        # self.agent_trajectory.append(np.concatenate((self.robots[0].get_position(), self.robots[0].get_orientation())))
        # self.object_trajectory.append([np.concatenate((obj.get_position(), obj.get_orientation()))
        #                                for obj in self.interactive_objects])
        self.path_length += np.linalg.norm(self.robots[0].get_position() - robot_position)
        # self.kinematic_disturbance += np.sum([
        #     obj.mass * np.linalg.norm(np.array(obj.get_position()) - np.array(prev_pos))
        #     for obj, prev_pos in zip(self.interactive_objects, object_positions)
        #     if obj.body_id in self.interactive_objects_collided
        # ])


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
        self.stage = 0
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
                reward += self.success_reward / 2.0
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
        # floor_height = 0.0 if self.floor_num is None else self.scene.get_floor_height(self.floor_num)
        # if self.robots[0].get_position()[2] > floor_height + self.death_z_thresh:
        #     reward -= self.success_reward * 1.0

        # push door the wrong way
        # door_angle = p.getJointState(self.door.body_id, self.door_axis_link_id)[0]
        # if door_angle > (10.0 / 180.0 * np.pi):
        #     reward -= self.success_reward * 1.0

        # print("get_reward (stage %d): %f" % (self.stage, reward))
        return reward, info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        help='which config file to use [default: use yaml files in examples/configs]', required=True)
    parser.add_argument('--env_type',
                        choices=['deterministic', 'random', 'interactive', 'ig', 'ig_s2r'],
                        default='deterministic',
                        help='which environment type (deterministic | random | interactive | ig', required=True)
    parser.add_argument('--mode', '-m',
                        choices=['headless', 'gui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    args = parser.parse_args()

    if args.env_type == 'deterministic':
        nav_env = NavigateEnv(config_file=args.config,
                              mode=args.mode,
                              action_timestep=1.0 / 10.0,
                              physics_timestep=1.0 / 40.0)
    elif args.env_type == 'random':
        nav_env = NavigateRandomEnv(config_file=args.config,
                                    mode=args.mode,
                                    action_timestep=1.0 / 10.0,
                                    physics_timestep=1.0 / 40.0)
    elif args.env_type == 'ig':
        nav_env = InteractiveGibsonNavigateEnv(config_file=args.config,
                                               mode=args.mode,
                                               action_timestep=1.0 / 10.0,
                                               physics_timestep=1.0 / 40.0)
    elif args.env_type == 'ig_s2r':
        nav_env = InteractiveGibsonNavigateSim2RealEnv(config_file=args.config,
                                                       mode=args.mode,
                                                       action_timestep=1.0 / 10.0,
                                                       physics_timestep=1.0 / 40.0)
    elif args.env_type == 'interactive':
        nav_env = InteractiveNavigateEnv(config_file=args.config,
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

    step_time_list = []
    for episode in range(100):
        nav_env.reset()
        print('Episode: {}'.format(episode))
        start = time.time()
        for step in range(50):  # 500 steps, 50s world time
            action = nav_env.action_space.sample()
            # action[:] = 1.0
            # action[0] = -0.4666666666666666
            # action[1] = -0.2
            # action[0] = 2.0 / 3.0
            # action[1] = -1.0
            # action[:] = 1.0
            # action[:] = -1.0 / 3
            # if nav_env.stage == 0:
            #     action[:2] = 0.5
            # elif nav_env.stage == 1:
            #     action[:2] = -0.5

            # action = np.zeros(nav_env.action_space.shape)
            # debug_param_values = [p.readUserDebugParameter(debug_param) for debug_param in debug_params]
            # action[2:] = np.array(debug_param_values)
            state, reward, done, _ = nav_env.step(action)
            #print('reward', reward)
            if done:
                print('Episode finished after {} timesteps'.format(step + 1))
                break
        episode_time = time.time() - start
        step_time = episode_time / 100.0
        fps = 1.0 / step_time
        print('episode_time', episode_time)
        print('step_time', step_time)
        print('fps', fps)
        if episode != 0:
            step_time_list.append(step_time)

    print('avg_step_time', np.mean(step_time_list))
    print('avg_fps', 1.0 / np.mean(step_time_list))
    nav_env.clean()
