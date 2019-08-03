from gibson2.core.physics.interactive_objects import VisualObject, InteractiveObj, BoxShape
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
import multiprocessing

# define navigation environments following Anderson, Peter, et al. 'On evaluation of embodied navigation agents.'
# arXiv preprint arXiv:1807.06757 (2018).
# https://arxiv.org/pdf/1807.06757.pdf


class NavigateEnv(BaseEnv):
    def __init__(
            self,
            config_file,
            mode='headless',
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            automatic_reset=False,
            device_idx=0,
    ):
        super(NavigateEnv, self).__init__(config_file=config_file, mode=mode, device_idx=device_idx)
        self.automatic_reset = automatic_reset

        # simulation
        self.mode = mode
        self.action_timestep = action_timestep
        self.physics_timestep = physics_timestep
        self.simulator.set_timestep(physics_timestep)
        self.simulator_loop = int(self.action_timestep / self.simulator.timestep)
        self.current_step = 0
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
        assert self.reward_type in ['dense', 'sparse', 'normalized_l2', 'l2', 'stage_sparse']

        self.success_reward = self.config.get('success_reward', 10.0)
        self.slack_reward = self.config.get('slack_reward', -0.01)
        self.death_z_thresh = self.config.get('death_z_thresh', 0.1)
        # self.death_z_thresh = 0.2

        # reward weight
        self.potential_reward_weight = self.config.get('potential_reward_weight', 10.0)
        self.electricity_reward_weight = self.config.get('electricity_reward_weight', 0.0)
        self.stall_torque_reward_weight = self.config.get('stall_torque_reward_weight', 0.0)
        self.collision_reward_weight = self.config.get('collision_reward_weight', 0.0)
        # ignore the agent's collision with these body ids, typically ids of the ground
        self.collision_ignore_body_ids = set(self.config.get('collision_ignore_body_ids', []))

        # discount factor
        self.discount_factor = self.config.get('discount_factor', 1.0)
        self.output = self.config['output']
        self.n_rays_per_horizontal = self.config.get('n_rays_per_horizontal', 128)

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
                                            shape=(self.config['resolution'],
                                                   self.config['resolution'], 3),
                                            dtype=np.float32)
            observation_space['rgb'] = self.rgb_space
        if 'depth' in self.output:
            self.depth_space = gym.spaces.Box(low=-np.inf,
                                              high=np.inf,
                                              shape=(self.config['resolution'],
                                                     self.config['resolution'], 1),
                                              dtype=np.float32)
            observation_space['depth'] = self.depth_space
        if 'scan' in self.output:
            self.scan_space = gym.spaces.Box(low=-np.inf,
                                             high=np.inf,
                                             shape=(self.n_rays_per_horizontal,),
                                             dtype=np.float32)
            observation_space['scan'] = self.scan_space
        if 'rgb_filled' in self.output:  # use filler
            self.comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
            self.comp = torch.nn.DataParallel(self.comp).cuda()
            self.comp.load_state_dict(
                torch.load(os.path.join(gibson2.assets_path, 'networks', 'model.pth')))
            self.comp.eval()

        self.observation_space = gym.spaces.Dict(observation_space)
        self.action_space = self.robots[0].action_space

        # variable initialization
        self.current_episode = 0

        # add visual objects
        self.visual_object_at_initial_target_pos = self.config.get(
            'visual_object_at_initial_target_pos', False)

        if self.visual_object_at_initial_target_pos:
            cyl_length = 3
            self.initial_pos_vis_obj = VisualObject(visual_shape=p.GEOM_CYLINDER,
                                                    rgba_color=[1, 0, 0, 0.95],
                                                    radius=0.5,
                                                    length=cyl_length,
                                                    initial_offset=[0, 0, cyl_length / 2.0])
            self.target_pos_vis_obj = VisualObject(visual_shape=p.GEOM_CYLINDER,
                                                   rgba_color=[0, 0, 1, 0.95],
                                                   radius=0.5,
                                                   length=cyl_length,
                                                   initial_offset=[0, 0, cyl_length / 2.0])
            self.initial_pos_vis_obj.load()
            if self.config.get('target_visual_object_visible_to_agent', False):
                self.simulator.import_object(self.target_pos_vis_obj)
            else:
                self.target_pos_vis_obj.load()



    def reload(self, config_file):
        super(NavigateEnv, self).reload(config_file)
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
        self.dist_tol = self.config.get('dist_tol', 0.5)
        self.max_step = self.config.get('max_step', float('inf'))

        # reward
        self.electricity_cost = self.config.get('electricity_cost', 0.0)
        self.stall_torque_cost = self.config.get('stall_torque_cost', 0.0)
        self.collision_cost = self.config.get('collision_cost', 0.0)
        self.discount_factor = self.config.get('discount_factor', 1.0)
        self.output = self.config['output']

        self.sensor_dim = self.additional_states_dim
        self.action_dim = self.robots[0].action_dim

        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.sensor_dim,), dtype=np.float64)
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
        if 'rgb' in self.output:
            self.rgb_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.config['resolution'],
                                                   self.config['resolution'], 3),
                                            dtype=np.float32)
            observation_space['rgb'] = self.rgb_space
        if 'depth' in self.output:
            self.depth_space = gym.spaces.Box(low=0.0,
                                              high=1.0,
                                              shape=(self.config['resolution'],
                                                     self.config['resolution'], 1),
                                              dtype=np.float32)
            observation_space['depth'] = self.depth_space
        if 'rgb_filled' in self.output:  # use filler
            self.comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
            self.comp = torch.nn.DataParallel(self.comp).cuda()
            self.comp.load_state_dict(
                torch.load(os.path.join(gibson2.assets_path, 'networks', 'model.pth')))
            self.comp.eval()
        if 'pointgoal' in self.output:
            observation_space['pointgoal'] = gym.spaces.Box(low=-np.inf,
                                                            high=np.inf,
                                                            shape=(2,),
                                                            dtype=np.float32)

        self.observation_space = gym.spaces.Dict(observation_space)
        self.action_space = self.robots[0].action_space

        self.visual_object_at_initial_target_pos = self.config.get(
            'visual_object_at_initial_target_pos', False)
        if self.visual_object_at_initial_target_pos:
            self.initial_pos_vis_obj = VisualObject(rgba_color=[1, 0, 0, 0.5])
            self.target_pos_vis_obj = VisualObject(rgba_color=[0, 0, 1, 0.5])
            self.initial_pos_vis_obj.load()
            if self.config.get('target_visual_object_visible_to_agent', False):
                self.simulator.import_object(self.target_pos_vis_obj)
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
        """
        relative_position = self.target_pos - self.robots[0].get_position()
        # rotate relative position back to body point of view
        relative_position_odom = rotate_vector_3d(relative_position, *self.robots[0].get_rpy())
        # the angle between the direction the agent is facing and the direction to the target position
        delta_yaw = np.arctan2(relative_position_odom[1], relative_position_odom[0])
        additional_states = np.concatenate((relative_position,
                                            relative_position_odom,
                                            [np.sin(delta_yaw), np.cos(delta_yaw)]))
        if self.config['task'] == 'reaching':
            # get end effector information

            end_effector_pos = self.robots[0].get_end_effector_position() - self.robots[0].get_position()
            end_effector_pos = rotate_vector_3d(end_effector_pos, *self.robots[0].get_rpy())
            additional_states = np.concatenate((additional_states, end_effector_pos))

        assert len(additional_states) == self.additional_states_dim, 'additional states dimension mismatch'
        return additional_states
        """

    def get_auxiliary_sensor(self, collision_links=[]):
        return np.array([])

    def get_state(self, collision_links=[]):
        # calculate state
        # sensor_state = self.robots[0].calc_state()
        # sensor_state = np.concatenate((sensor_state, self.get_additional_states()))
        sensor_state = self.get_additional_states()
        auxiliary_sensor = self.get_auxiliary_sensor(collision_links)

        # rgb = self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
        # depth = np.clip(depth, 0.0, 5.0) / 5.0
        # depth = 1.0 - depth  # flip black/white
        # cv2.imshow('rgb', rgb)
        # cv2.imshow('depth', depth)

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
            state['seg'] = self.simulator.renderer.render_robot_cameras(modes='seg')
        if 'rgb_filled' in self.output:
            with torch.no_grad():
                tensor = transforms.ToTensor()((state['rgb'] * 255).astype(np.uint8)).cuda()
                rgb_filled = self.comp(tensor[None, :, :, :])[0].permute(1, 2, 0).cpu().numpy()
                state['rgb_filled'] = rgb_filled

        if 'pointgoal' in self.output:
            state['pointgoal'] = sensor_state[:2]

        if 'scan' in self.output:
            assert 'scan_link' in self.robots[0].parts, "Requested scan but no scan_link"
            pose_camera = self.robots[0].parts['scan_link'].get_pose()

            n_vertical_beams = 1
            angle = np.arange(0, 2 * np.pi, 2 * np.pi / float(self.n_rays_per_horizontal))
            elev_bottom_angle = 0
            elev_top_angle = 0
            elev_angle = [0]
            orig_offset = np.vstack([
                np.vstack([np.cos(angle),
                           np.sin(angle),
                           np.repeat(np.tan(elev_ang), angle.shape)]).T for elev_ang in elev_angle
            ])
            transform_matrix = quat2mat(
                [pose_camera[-1], pose_camera[3], pose_camera[4], pose_camera[5]])
            offset = orig_offset.dot(np.linalg.inv(transform_matrix))
            pose_camera = pose_camera[None, :3].repeat(self.n_rays_per_horizontal * n_vertical_beams,
                                                       axis=0)

            results = p.rayTestBatch(pose_camera, pose_camera + offset * 30)
            hit = np.array([item[0] for item in results])
            dist = np.array([item[2] for item in results])
            #dist[dist >= 1 - 1e-5] = np.nan
            #dist[dist < 0.1 / 30] = np.nan
            dist *= 30
            dist[hit == self.robots[0].robot_ids[0]] = -1
            #dist[hit == -1] = np.nan

            #xyz = dist[:, np.newaxis] * orig_offset
            #xyz = xyz[np.equal(np.isnan(xyz), False)]  # Remove nans
            # print(xyz.shape)
            #xyz = xyz.reshape(xyz.shape[0] // 3, -1)
            state['scan'] = dist
            # dist = np.clip(dist, 0.0, 5.0) / 5.0
            # cv2.imshow('scan', dist)

        return state

    def run_simulation(self):
        collision_links = []
        for _ in range(self.simulator_loop):
            self.simulator_step()
            collision_links += list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0]))
        return self.filter_collision_links(collision_links)

    def filter_collision_links(self, collision_links):
        return [elem for elem in collision_links if elem[2] not in self.collision_ignore_body_ids]

    def get_position_of_interest(self):
        if self.config['task'] == 'pointgoal':
            return self.robots[0].get_position()
        elif self.config['task'] == 'reaching':
            return self.robots[0].get_end_effector_position()

    def get_potential(self):
        return l2_distance(self.target_pos, self.get_position_of_interest())

    def get_reward(self, collision_links=[], action=None, info={}):
        reward = self.slack_reward  # |slack_reward| = 0.01 per step

        new_normalized_potential = self.get_potential() / self.initial_potential

        potential_reward = self.normalized_potential - new_normalized_potential
        reward += potential_reward * self.potential_reward_weight  # |potential_reward| ~= 0.1 per step
        self.normalized_potential = new_normalized_potential

        # electricity_reward = np.abs(self.robots[0].joint_speeds * self.robots[0].joint_torque).mean().item()
        electricity_reward = 0.0
        reward += electricity_reward * self.electricity_reward_weight  # |electricity_reward| ~= 0.05 per step

        # stall_torque_reward = np.square(self.robots[0].joint_torque).mean()
        stall_torque_reward = 0.0
        reward += stall_torque_reward * self.stall_torque_reward_weight  # |stall_torque_reward| ~= 0.05 per step

        collision_reward = float(len(collision_links) > 0)
        info['collision_reward'] = collision_reward * self.collision_reward_weight  # expose collision reward to info
        reward += collision_reward * self.collision_reward_weight  # |collision_reward| ~= 1.0 per step if collision

        # goal reached
        if l2_distance(self.target_pos, self.get_position_of_interest()) < self.dist_tol:
            reward += self.success_reward  # |success_reward| = 10.0 per step

        return reward, info

    def get_termination(self, collision_links=[], info={}):
        self.current_step += 1
        done = False

        # for elem in collision_links:
        #     if elem[9] > 500:
        #         print("collision between " + self.id_to_name[self.robots[0].robot_ids[0]]["links"][elem[3]]
        #               + " and " + self.id_to_name[elem[2]]["links"][elem[4]])

        # door_angle = p.getJointState(self.door.body_id, self.door_axis_link_id)[0]
        # max_force = max([elem[9] for elem in collision_links]) if len(collision_links) > 0 else 0

        if l2_distance(self.target_pos, self.get_position_of_interest()) < self.dist_tol:
            print("GOAL")
            done = True
            info['success'] = True

        elif self.robots[0].get_position()[2] > self.death_z_thresh:
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
            info['episode_length'] = self.current_step
            info['collision_step'] = self.collision_step
            info['energy_cost'] = self.energy_cost
            info['stage'] = self.stage

        return done, info

    def step(self, action):
        self.robots[0].apply_action(action)
        collision_links = self.run_simulation()
        state = self.get_state(collision_links)
        info = {}
        reward, info = self.get_reward(collision_links, action, info)
        done, info = self.get_termination(collision_links, info)

        if done and self.automatic_reset:
            info['last_observation'] = state
            state = self.reset()
        return state, reward, done, info

    def reset_initial_and_target_pos(self):
        self.robots[0].set_position(pos=self.initial_pos)
        self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(*self.initial_orn), 'wxyz'))

    def reset(self):
        self.current_episode += 1
        self.robots[0].robot_specific_reset()
        self.reset_initial_and_target_pos()
        self.initial_potential = self.get_potential()
        self.normalized_potential = 1.0
        if self.reward_type == 'normalized_l2':
            self.initial_l2_potential = self.get_l2_potential()
            self.normalized_l2_potential = 1.0
        elif self.reward_type == 'l2':
            self.l2_potential = self.get_l2_potential()
        elif self.reward_type == 'dense':
            self.potential = self.get_potential()

        self.current_step = 0
        self.collision_step = 0
        self.energy_cost = 0.0

        # set position for visual objects
        if self.visual_object_at_initial_target_pos:
            self.initial_pos_vis_obj.set_position(self.initial_pos)
            self.target_pos_vis_obj.set_position(self.target_pos)

        state = self.get_state()
        return state


class NavigateRandomEnv(NavigateEnv):
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
        super(NavigateRandomEnv, self).__init__(config_file,
                                                mode=mode,
                                                action_timestep=action_timestep,
                                                physics_timestep=physics_timestep,
                                                automatic_reset=automatic_reset,
                                                device_idx=device_idx)
        self.random_height = random_height

    def reset_initial_and_target_pos(self):
        num_trials = 0
        max_trials = 10
        while True:  # if collision happens, reinitialize
            floor, pos = self.scene.get_random_point()
            self.robots[0].set_position(pos=[pos[0], pos[1], pos[2] + 0.1])
            self.robots[0].set_orientation(
                orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz'))

            collision_links = []
            for _ in range(self.simulator_loop):
                self.simulator_step()
                collision_links += list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0]))
            collision_links = self.filter_collision_links(collision_links)
            self.initial_pos = pos

            if len(collision_links) == 0:
                break

            num_trials += 1
            if num_trials == max_trials:
                raise Exception("Failed to initialize agent's position: collision occurs for %d times." % (max_trials))

        dist = 0.0
        while dist < 1.0:  # if initial and target positions are < 1 meter away from each other, reinitialize
            _, self.target_pos = self.scene.get_random_point_floor(floor, self.random_height)
            dist = l2_distance(self.initial_pos, self.target_pos)


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

        self.floor = VisualObject(visual_shape=p.GEOM_BOX, rgba_color=[0.643, 0.643, 0.788, 0.0], half_extents=[20, 20, 0.02], initial_offset=[0, 0, -0.03])
        self.floor.load()
        self.floor.set_position([0, 0, 0])
        self.simulator.import_object(self.floor)

        self.door = InteractiveObj(os.path.join(gibson2.assets_path, 'models', 'scene_components', 'realdoor.urdf'),
                                   scale=1.35)
        self.arena = arena
        self.simulator.import_interactive_object(self.door)
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

        assert self.arena in [
            "only_ll_obstacles",
            "only_ll",
            "simple_hl_ll",
            "complex_hl_ll"
        ], "Wrong arena"
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
                box = BoxShape(pos=box_pose[0], dim=[0.5,0.5,1])
                self.simulator.import_interactive_object(box)
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
                self.simulator.import_interactive_object(wall)
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
                self.simulator.import_interactive_object(wall)
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
                self.simulator.import_interactive_object(wall)
                wall.set_position_rotation(wall_pose[0], wall_pose[1])
                self.walls += [wall]

            for wall_pose in self.half_wall_poses:
                wall = InteractiveObj(
                    os.path.join(gibson2.assets_path, 'models', 'scene_components', 'walls_half.urdf'),
                    scale=1)
                self.simulator.import_interactive_object(wall)
                wall.set_position_rotation(wall_pose[0], wall_pose[1])
                self.walls += [wall]

            for wall_pose in self.quarter_wall_poses:
                wall = InteractiveObj(
                    os.path.join(gibson2.assets_path, 'models', 'scene_components', 'walls_quarter.urdf'),
                    scale=1)
                self.simulator.import_interactive_object(wall)
                wall.set_position_rotation(wall_pose[0], wall_pose[1])
                self.walls += [wall]

        else:
            print("Wrong arena")
            exit(-1)

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
        # self.subgoal_base = VisualObject(visual_shape=p.GEOM_BOX, rgba_color=[0, 1, 0, 0.5], half_extents=[0.4] * 3)
        # self.subgoal_base.load()
        self.subgoal_end_effector = VisualObject(rgba_color=[0, 0, 0, 0.8], radius=0.06)
        self.subgoal_end_effector.load()

        self.subgoal_end_effector_base = VisualObject(visual_shape=p.GEOM_CYLINDER,
                                                      rgba_color=[1, 1, 0, 0.8],
                                                      radius=0.05,
                                                      length=3,
                                                      initial_offset=[0, 0, 3.0 / 2])
        self.subgoal_end_effector_base.load()

        self.door_handle_vis = VisualObject(rgba_color=[1, 0, 0, 0.0], radius=self.door_handle_dist_thresh)
        self.door_handle_vis.load()
        # self.door_vis = VisualObject(visual_shape=p.GEOM_BOX, rgba_color=[0, 1, 1, 0.5], half_extents=[0.05, 0.5, 2.7])
        # self.door_vis.load()

        self.id_to_name = {
            0: {"name": "ground", "links": {-1: "base", 0: "ground"}},
            1: {"name": "ground", "links": {-1: "base", 0: "ground"}},
            2: {"name": "ground", "links": {-1: "base", 0: "ground"}},
            3: {"name": "ground", "links": {-1: "base", 0: "ground"}},
        }
        self.id_to_name[self.door.body_id] = {"name": "door", "links": {-1: "world", 0: "base", 1: "door_leaf", 2: "door_knob"}}
        for i, wall in enumerate(self.walls):
            self.id_to_name[wall.body_id] = {"name": "wall%d" % (i+1), "links": {-1: "world", 0: "wall"}}
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

    def set_subgoal(self, ideal_next_state):
        obs_avg = (self.observation_normalizer['sensor'][1] + self.observation_normalizer['sensor'][0]) / 2.0
        obs_mag = (self.observation_normalizer['sensor'][1] - self.observation_normalizer['sensor'][0]) / 2.0
        ideal_next_state = (ideal_next_state * obs_mag) + obs_avg
        # ideal_next_state = self.wrap_to_pi(ideal_next_state, np.array([5, 6]))
        #
        # base_pos = np.zeros(3)
        # z = self.robots[0].get_position()[2]
        # base_pos[:2] = ideal_next_state[:2]
        # base_pos[2] = z
        #
        # yaw = ideal_next_state[5]
        # new_orn = quatToXYZW(euler2quat(0, 0, yaw), 'wxyz')
        #
        # end_effector_pos = ideal_next_state[2:5]
        # roll, pitch, _ = self.robots[0].get_rpy()
        # end_effector_pos = rotate_vector_3d(end_effector_pos, -roll, -pitch, -yaw)
        # # end_effector_pos = rotate_vector_3d(end_effector_pos, 0, 0, -yaw)
        # # print("end effector pos", end_effector_pos)
        #
        # self.subgoal_base.set_position(np.array([base_pos[0], base_pos[1], base_pos[2] + 0.4]), new_orn=new_orn)
        # self.subgoal_end_effector.set_position(base_pos + end_effector_pos)
        #
        # door_angle = ideal_next_state[6]
        # door_width = 0.5
        # door_x = -np.sin(door_angle) * door_width
        # door_y = (np.cos(door_angle) - 1) * door_width
        # door_orn = quatToXYZW(euler2quat(0, 0, door_angle), 'wxyz')
        # self.door_vis.set_position(np.array([door_x, door_y, 0.0]), new_orn=door_orn)
        self.subgoal_end_effector.set_position(ideal_next_state)
        self.subgoal_end_effector_base.set_position([ideal_next_state[0], ideal_next_state[1], 0])

    def set_subgoal_color(self, rgba_color=[1, 0, 0, 0.5]):
        self.subgoal_end_effector.set_color(rgba_color)

    def set_subgoal_type(self, only_base=True):
        if only_base:
            # Make the marker for the end effector completely transparent
            self.subgoal_end_effector.set_color([0, 0, 0, 0.0])
        else:
            self.subgoal_end_effector.set_color([0, 0, 0, 0.8])

    def reset_interactive_objects(self):
        # p.resetJointState(self.door.body_id, self.door_axis_link_id, targetValue=(100.0 / 180.0 * np.pi), targetVelocity=0.0)
        p.resetJointState(self.door.body_id, self.door_axis_link_id, targetValue=0.0, targetVelocity=0.0)
        if self.cid is not None:
            p.removeConstraint(self.cid)
            self.cid = None

    def reset_initial_and_target_pos(self):
        num_trials = 0
        max_trials = 10

        while True:  # if collision happens, reinitialize
            # pos = [np.random.uniform(1, 2), np.random.uniform(-0.5, 0.5), 0]
            if self.arena == "only_ll" or self.arena == "only_ll_obstacles":
                # pos = [0.0, 0.0, 0.0]
                pos = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), 0]
            elif self.arena == "simple_hl_ll":
                if self.random_position:
                    pos = [np.random.uniform(1, 2), np.random.uniform(-2, 2), 0]
                    # pos = [np.random.uniform(0.5, 1.5), np.random.uniform(1.5, 2.0), 0]
                else:
                    pos = [1.0, 0.0, 0.0]
                    # pos = [1.0, 2.0, 0.0]
            elif self.arena == "complex_hl_ll":
                if self.random_position:
                    pos = [np.random.uniform(-2, -1.7), np.random.uniform(4.5, 5), 0]
                else:
                    pos = [-2, 4, 0.0]
                # pos = [np.random.uniform(11, 13), np.random.uniform(-2, 2), 0]
            else:
                print("Wrong ARENA name")
                exit(-1)

            # self.robots[0].set_position(pos=[pos[0], pos[1], pos[2] + 0.1])
            self.robots[0].set_position(pos=[pos[0], pos[1], pos[2]])

            if self.arena == "only_ll" or self.arena == "only_ll_obstacles":
                self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz'))
            elif self.arena == "simple_hl_ll":
                if self.random_position:
                    self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz'))
                else:
                    self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(0, 0, np.pi), 'wxyz'))
                    # self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(0, 0, -np.pi / 2), 'wxyz'))
            elif self.arena == "complex_hl_ll":
                self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz'))
                # if self.random_position:
                #     self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz'))
                # else:
                #     self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(0, 0, 0), 'wxyz'))
            else:
                print("Wrong ARENA name")
                exit(-1)

            collision_links = []
            for _ in range(self.simulator_loop):
                self.simulator_step()
                collision_links += list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0]))
            collision_links = self.filter_collision_links(collision_links)
            self.initial_pos = pos

            if len(collision_links) == 0:
                break

            num_trials += 1
            if num_trials == max_trials:
                raise Exception("Failed to initialize agent's position: collision occurs for %d times." % (max_trials))

        # # wait for the base to fall down to the ground and for the arm to move to its initial position
        # for _ in range(int(0.5 / self.physics_timestep)):
        #     self.simulator_step()

        if self.arena == "only_ll" or self.arena == "only_ll_obstacles":
            # self.target_pos = [-100, -100, 0]
            self.target_pos = [np.random.uniform(-200, -199), np.random.uniform(-2, 2), 0.0]
        elif self.arena == "simple_hl_ll" or self.arena == "complex_hl_ll":
            if self.random_position:
                self.target_pos = [np.random.uniform(-2, -1), np.random.uniform(-2, 2), 0.0]
            else:
                self.target_pos = np.array([-1.5, 0.0, 0.0])

        self.door_handle_vis.set_position(pos=np.array(p.getLinkState(self.door.body_id, self.door_handle_link_id)[0]))

    def reset(self):
        self.reset_interactive_objects()
        self.stage = 0
        # self.stage = self.stage_get_to_target_pos
        self.prev_stage = self.stage
        return super(InteractiveNavigateEnv, self).reset()

    def wrap_to_pi(self, states, indices):
        states[indices] = states[indices] - np.pi * 2 * np.floor((states[indices] + np.pi) / (np.pi * 2))
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
        # robot_position = self.robots[0].get_position()[:2]  # z is not controllable by the agent
        # end_effector_pos = self.robots[0].get_end_effector_position() - self.robots[0].get_position()
        # end_effector_pos = rotate_vector_3d(end_effector_pos, *self.robots[0].get_rpy())
        # # print("end_effector_pos", end_effector_pos)
        # # print("joint", p.getJointState(self.robots[0].robot_ids[0], 27)[0])
        # _, _, yaw = self.robots[0].get_rpy()
        # door_angle = p.getJointState(self.door.body_id, self.door_axis_link_id)[0]
        # additional_states = np.concatenate([robot_position, end_effector_pos, [yaw, door_angle]])
        # additional_states = self.wrap_to_pi(additional_states, np.array([5, 6]))
        # assert len(additional_states) == self.additional_states_dim, 'additional states dimension mismatch'
        # return additional_states

        additional_states = self.robots[0].get_end_effector_position()
        assert len(additional_states) == self.additional_states_dim, 'additional states dimension mismatch'
        return additional_states

    def get_auxiliary_sensor(self, collision_links=[]):
        auxiliary_sensor = np.zeros(self.auxiliary_sensor_dim)
        robot_state = self.robots[0].calc_state()
        # assert self.auxiliary_sensor_dim == 44
        # assert self.auxiliary_sensor_dim == 58
        assert self.auxiliary_sensor_dim == 66
        assert robot_state.shape[0] == 31

        robot_state = self.wrap_to_pi(robot_state, np.arange(7, 28, 3))  # wrap wheel and arm joint pos to [-pi, pi]

        # auxiliary_sensor[:6] = robot_state[:6]                   # z, vx, vy, vz, roll, pitch
        # auxiliary_sensor[6:41:5] = robot_state[7:28:3]           # pos for wheel 1, 2, arm joint 1, 2, 3, 4, 5
        # auxiliary_sensor[7:42:5] = robot_state[8:29:3]           # vel for wheel 1, 2, arm joint 1, 2, 3, 4, 5
        # auxiliary_sensor[8:43:5] = robot_state[9:30:3]           # trq for wheel 1, 2, arm joint 1, 2, 3, 4, 5
        # auxiliary_sensor[9:44:5] = np.cos(robot_state[7:28:3])   # cos(pos) for wheel 1, 2, arm joint 1, 2, 3, 4, 5
        # auxiliary_sensor[10:45:5] = np.sin(robot_state[7:28:3])  # sin(pos) for wheel 1, 2, arm joint 1, 2, 3, 4, 5
        # auxiliary_sensor[41:44] = robot_state[37:40]             # v_roll, v_pitch, v_yaw

        # auxiliary_sensor[:6] = robot_state[:6]        # z, vx, vy, vz, roll, pitch
        # auxiliary_sensor[6:27] = robot_state[7:28]    # wheel 1, 2, arm joint 1, 2, 3, 4, 5
        # auxiliary_sensor[27:30] = robot_state[37:40]  # v_roll, v_pitch, v_yaw

        # roll, pitch, yaw = self.robots[0].get_rpy()
        # cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        # door_angle = p.getJointState(self.door.body_id, self.door_axis_link_id)[0]
        # cos_door_angle, sin_door_angle = np.cos(door_angle), np.sin(door_angle)
        # has_door_handle_in_hand = 1.0 if self.stage == self.stage_open_door else -1.0
        # door_pos = np.array([0, 0, -0.02])
        # target_pos = self.target_pos
        # robot_pos = self.robots[0].get_position()
        # door_pos_local = rotate_vector_3d(door_pos - robot_pos, roll, pitch, yaw)
        # target_pos_local = rotate_vector_3d(target_pos - robot_pos, roll, pitch, yaw)

        # auxiliary_sensor[30:32] = np.array([cos_yaw, sin_yaw])
        # auxiliary_sensor[32:35] = np.array([cos_door_angle, sin_door_angle, has_door_handle_in_hand])
        # auxiliary_sensor[35:38] = target_pos
        # auxiliary_sensor[38:41] = door_pos_local
        # auxiliary_sensor[41:44] = target_pos_local

        # auxiliary_sensor[44:46] = np.array([cos_yaw, sin_yaw])
        # auxiliary_sensor[46:49] = np.array([cos_door_angle, sin_door_angle, has_door_handle_in_hand])
        # auxiliary_sensor[49:52] = target_pos
        # auxiliary_sensor[52:55] = door_pos_local
        # auxiliary_sensor[55:58] = target_pos_local

        end_effector_pos = self.robots[0].get_end_effector_position() - self.robots[0].get_position()
        end_effector_pos = rotate_vector_3d(end_effector_pos, *self.robots[0].get_rpy())
        auxiliary_sensor[:3] = self.robots[0].get_position()     # x, y, z
        auxiliary_sensor[3:6] = end_effector_pos                 # arm_x, arm_y_ arm_z (local)
        auxiliary_sensor[6:11] = robot_state[1:6]                # vx, vy, vz, roll, pitch
        auxiliary_sensor[11:46:5] = robot_state[7:28:3]          # pos for wheel 1, 2, arm joint 1, 2, 3, 4, 5
        auxiliary_sensor[12:47:5] = robot_state[8:29:3]          # vel for wheel 1, 2, arm joint 1, 2, 3, 4, 5
        auxiliary_sensor[13:48:5] = robot_state[9:30:3]          # trq for wheel 1, 2, arm joint 1, 2, 3, 4, 5
        auxiliary_sensor[14:49:5] = np.cos(robot_state[7:28:3])  # cos(pos) for wheel 1, 2, arm joint 1, 2, 3, 4, 5
        auxiliary_sensor[15:50:5] = np.sin(robot_state[7:28:3])  # sin(pos) for wheel 1, 2, arm joint 1, 2, 3, 4, 5
        auxiliary_sensor[46:49] = robot_state[28:31]             # v_roll, v_pitch, v_yaw

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
        has_collision = 1.0 if len(collision_links) > 0 else -1.0

        auxiliary_sensor[49:52] = np.array([yaw, cos_yaw, sin_yaw])
        auxiliary_sensor[52:56] = np.array([door_angle, cos_door_angle, sin_door_angle, has_door_handle_in_hand])
        auxiliary_sensor[56:59] = target_pos
        auxiliary_sensor[59:62] = door_pos_local
        auxiliary_sensor[62:65] = target_pos_local
        auxiliary_sensor[65] = has_collision

        return auxiliary_sensor

    def filter_collision_links(self, collision_links):
        collision_links = [elem for elem in collision_links if elem[2] not in self.collision_ignore_body_ids]
        # ignore collision between hand and door
        collision_links = [elem for elem in collision_links if
                           not (elem[2] == self.door.body_id and elem[3] in [32, 33])]
        return collision_links

    def step(self, action):
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

        if self.stage == self.stage_open_door and p.getJointState(self.door.body_id, 1)[0] > self.door_angle:  # door open > 45/60/90 degree
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

    def get_l2_potential(self):
        return l2_distance(self.target_pos, self.get_position_of_interest())

    def get_reward(self, collision_links=[], action=None, info={}):
        reward = 0.0

        if self.reward_type == 'dense':
            if self.stage != self.prev_stage:
                # advance to the next stage
                # self.initial_potential = self.get_potential()
                # self.normalized_potential = 1.0
                self.potential = self.get_potential()
                reward += self.success_reward / 2.0
            else:
                # new_normalized_potential = self.get_potential() / self.initial_potential
                # potential_reward = self.normalized_potential - new_normalized_potential
                # reward += potential_reward * self.potential_reward_weight  # |potential_reward| ~= 0.1 per step
                # # self.reward_stats.append(np.abs(potential_reward * self.potential_reward_weight))
                # self.normalized_potential = new_normalized_potential
                new_potential = self.get_potential()
                potential_reward = self.potential - new_potential

                reward += potential_reward * self.potential_reward_weight  # |potential_reward| ~= 0.1 per step
                # if self.stage == self.stage_open_door:
                #     print("Door stage reward ", reward)
                self.potential = new_potential
        elif self.reward_type == 'normalized_l2':
            new_normalized_l2_potential = self.get_l2_potential() / self.initial_l2_potential
            potential_reward = self.normalized_l2_potential - new_normalized_l2_potential
            reward += potential_reward * self.potential_reward_weight
            self.normalized_l2_potential = new_normalized_l2_potential
        elif self.reward_type == "l2":
            new_l2_potential = self.get_l2_potential()
            potential_reward = self.l2_potential - new_l2_potential
            reward += potential_reward * self.potential_reward_weight
            self.l2_potential = new_l2_potential
        elif self.reward_type == 'stage_sparse':
            if self.stage != self.prev_stage:
                reward += self.success_reward / 2.0

        base_moving = np.any(np.abs(action[:2]) >= 0.01)
        arm_moving = np.any(np.abs(action[2:]) >= 0.01)
        electricity_reward = float(base_moving) + float(arm_moving)
        self.energy_cost += electricity_reward
        reward += electricity_reward * self.electricity_reward_weight

        # electricity_reward = np.abs(self.robots[0].joint_speeds * self.robots[0].joint_torque).mean().item()
        # # electricity_reward = 0.0
        # reward += np.clip(electricity_reward * self.electricity_reward_weight, -0.005, 0)  # |electricity_reward| ~= 0.005 per step
        # # self.reward_stats.append(np.abs(electricity_reward * self.electricity_reward_weight))
        #
        # stall_torque_reward = np.square(self.robots[0].joint_torque).mean()
        # # stall_torque_reward = 0.0
        # reward += np.clip(stall_torque_reward * self.stall_torque_reward_weight, -0.005, 0)  # |stall_torque_reward| ~= 0.005 per step

        # collisions = [[elem[3], elem[2], elem[4]] for elem in collision_links
        #               if elem[3] in self.collision_links and not (elem[2] == self.door.body_id and elem[4] == self.door_handle_link_id)]
        # print('-' * 30)
        # for col in collisions:
        #     print('link a', self.id_to_name[self.robots[0].robot_ids[0]]["links"][col[0]],
        #           'body b', self.id_to_name[col[1]]["name"],
        #           'link b', self.id_to_name[col[1]]["links"][col[2]])

        collision_reward = float(len(collision_links) > 0)
        # print('collision_reward', collision_reward)
        self.collision_step += int(collision_reward)
        reward += collision_reward * self.collision_reward_weight  # |collision_reward| ~= 1.0 per step if collision
        info['collision_reward'] = collision_reward * self.collision_reward_weight  # expose collision reward to info

        # self.reward_stats.append(np.abs(collision_reward * self.collision_reward_weight))

        # goal reached
        if l2_distance(self.target_pos, self.get_position_of_interest()) < self.dist_tol:
            reward += self.success_reward  # |success_reward| = 10.0

        # death penalty
        if self.robots[0].get_position()[2] > self.death_z_thresh:
            reward -= self.success_reward * 1.0

        # push door the wrong way
        # door_angle = p.getJointState(self.door.body_id, self.door_axis_link_id)[0]
        # if door_angle > (10.0 / 180.0 * np.pi):
        #     reward -= self.success_reward * 1.0

        # print("get_reward (stage %d): %f" % (self.stage, reward))
        return reward, info


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
                        choices=['deterministic', 'random', 'interactive'],
                        default='deterministic',
                        help='which environment type (deterministic | random | interactive')
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
                              physics_timestep=1 / 40.0)
    elif args.env_type == 'random':
        nav_env = NavigateRandomEnv(config_file=config_filename,
                                    mode=args.mode,
                                    action_timestep=1.0 / 10.0,
                                    physics_timestep=1 / 40.0)
    else:
        nav_env = InteractiveNavigateEnv(config_file=config_filename,
                                         mode=args.mode,
                                         action_timestep=1.0 / 10.0,
                                         random_position=False,
                                         physics_timestep=1 / 40.0)

    # debug_params = [
    #     p.addUserDebugParameter('link1', -1.0, 1.0, -0.5),
    #     p.addUserDebugParameter('link2', -1.0, 1.0, 0.5),
    #     p.addUserDebugParameter('link3', -1.0, 1.0, 0.5),
    #     p.addUserDebugParameter('link4', -1.0, 1.0, 0.5),
    #     p.addUserDebugParameter('link5', -1.0, 1.0, 0.0),
    # ]
    # nav_env.reset()
    # for i in range(1000000):  # 500 steps, 50s world time
    #     debug_param_values = [p.readUserDebugParameter(debug_param) for debug_param in debug_params]
    #     action = np.zeros(nav_env.action_space.shape)
    #     action[2:] = np.array(debug_param_values)
    #     print(action)
    #     nav_env.step(action)
    # assert False

    for episode in range(50):
        print('Episode: {}'.format(episode))
        start = time.time()
        nav_env.reset()
        for i in range(500):  # 500 steps, 50s world time
            action = nav_env.action_space.sample()
            action[:] = 0
            # if nav_env.stage == 0:
            #     action[:2] = 0.5
            # elif nav_env.stage == 1:
            #     action[:2] = -0.5
            # if nav_env.stage == 1:
            #     action[:2] = -0.1

            state, reward, done, _ = nav_env.step(action)
            print(reward)
            # print(nav_env.stage)
            # embed()
            if done:
                print('Episode finished after {} timesteps'.format(i + 1))
                break
        print("episode time", time.time() - start)
        # print('len', len(nav_env.reward_stats))
        # print('mean', np.mean(nav_env.reward_stats))
        # print('median', np.median(nav_env.reward_stats))
        # print('max', np.max(nav_env.reward_stats))
        # print('min', np.min(nav_env.reward_stats))
        # print('std', np.std(nav_env.reward_stats))
        # print('95 percentile', np.percentile(nav_env.reward_stats, 95))
        # print('99 percentile', np.percentile(nav_env.reward_stats, 99))
    embed()

    nav_env.clean()
