import gibson2
from gibson2.utils.utils import quatToXYZW
from gibson2.envs.env_base import BaseEnv
from gibson2.tasks.room_rearrangement_task import RoomRearrangementTask
from gibson2.tasks.point_nav_fixed_task import PointNavFixedTask
from gibson2.tasks.point_nav_random_task import PointNavRandomTask
from gibson2.tasks.interactive_nav_random_task import InteractiveNavRandomTask
from gibson2.tasks.dynamic_nav_random_task import DynamicNavRandomTask
from gibson2.tasks.reaching_random_task import ReachingRandomTask

from transforms3d.euler import euler2quat
from collections import OrderedDict
import argparse
from transforms3d.quaternions import quat2mat
import gym
import numpy as np
import os
import pybullet as p
import time
import logging


class iGibsonEnv(BaseEnv):
    """
    iGibson Environment (OpenAI Gym interface)
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless or gui mode
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        super(iGibsonEnv, self).__init__(config_file=config_file,
                                         scene_id=scene_id,
                                         mode=mode,
                                         action_timestep=action_timestep,
                                         physics_timestep=physics_timestep,
                                         device_idx=device_idx,
                                         render_to_tensor=render_to_tensor)
        self.automatic_reset = automatic_reset

    def load_task_setup(self):
        """
        Load task setup
        """
        self.initial_pos_z_offset = self.config.get(
            'initial_pos_z_offset', 0.1)
        check_collision_distance = self.initial_pos_z_offset * 0.5
        # s = 0.5 * G * (t ** 2)
        check_collision_distance_time = np.sqrt(
            check_collision_distance / (0.5 * 9.8))
        self.check_collision_loop = int(
            check_collision_distance_time / self.action_timestep)

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(
            self.config.get('collision_ignore_body_b_ids', []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(
            self.config.get('collision_ignore_link_a_ids', []))

        # discount factor
        self.discount_factor = self.config.get('discount_factor', 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get(
            'texture_randomization_freq', None)
        self.object_randomization_freq = self.config.get(
            'object_randomization_freq', None)

        # task
        if self.config['task'] == 'point_nav_fixed':
            self.task = PointNavFixedTask(self)
        elif self.config['task'] == 'point_nav_random':
            self.task = PointNavRandomTask(self)
        elif self.config['task'] == 'interactive_nav_random':
            self.task = InteractiveNavRandomTask(self)
        elif self.config['task'] == 'dynamic_nav_random':
            self.task = DynamicNavRandomTask(self)
        elif self.config['task'] == 'reaching_random':
            self.task = ReachingRandomTask(self)
        elif self.config['task'] == 'room_rearrangement':
            self.task = RoomRearrangementTask(self)
        else:
            self.task = None

    def load_observation_space(self):
        """
        Load observation space
        """
        self.output = self.config['output']
        self.image_width = self.config.get('image_width', 128)
        self.image_height = self.config.get('image_height', 128)
        observation_space = OrderedDict()
        if 'task_obs' in self.output:
            self.task_obs_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.task.task_obs_dim,),
                dtype=np.float32)
            observation_space['task_obs'] = self.task_obs_space
        if 'rgb' in self.output:
            self.rgb_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.image_height,
                                                   self.image_width, 3),
                                            dtype=np.float32)
            observation_space['rgb'] = self.rgb_space
        if 'depth' in self.output:
            self.depth_noise_rate = self.config.get('depth_noise_rate', 0.0)
            self.depth_low = self.config.get('depth_low', 0.5)
            self.depth_high = self.config.get('depth_high', 5.0)
            self.depth_space = gym.spaces.Box(low=0.0,
                                              high=1.0,
                                              shape=(self.image_height,
                                                     self.image_width, 1),
                                              dtype=np.float32)
            observation_space['depth'] = self.depth_space
        if 'rgbd' in self.output:
            self.rgbd_space = gym.spaces.Box(low=0.0,
                                             high=1.0,
                                             shape=(self.image_height,
                                                    self.image_width, 4),
                                             dtype=np.float32)
            observation_space['rgbd'] = self.rgbd_space
        if 'seg' in self.output:
            self.seg_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.image_height,
                                                   self.image_width, 1),
                                            dtype=np.float32)
            observation_space['seg'] = self.seg_space
        if 'scan' in self.output:
            self.scan_noise_rate = self.config.get('scan_noise_rate', 0.0)
            self.n_horizontal_rays = self.config.get('n_horizontal_rays', 128)
            self.n_vertical_beams = self.config.get('n_vertical_beams', 1)
            assert self.n_vertical_beams == 1, 'scan can only handle one vertical beam for now'
            self.laser_linear_range = self.config.get(
                'laser_linear_range', 10.0)
            self.laser_angular_range = self.config.get(
                'laser_angular_range', 180.0)
            self.min_laser_dist = self.config.get('min_laser_dist', 0.05)
            self.laser_link_name = self.config.get(
                'laser_link_name', 'scan_link')
            self.scan_space = gym.spaces.Box(low=0.0,
                                             high=1.0,
                                             shape=(self.n_horizontal_rays *
                                                    self.n_vertical_beams, 1),
                                             dtype=np.float32)
            observation_space['scan'] = self.scan_space
        if 'rgb_filled' in self.output:  # use filler
            try:
                import torch.nn as nn
                import torch
                from torchvision import datasets, transforms
                from gibson2.learn.completion import CompletionNet
            except:
                raise Exception(
                    'Trying to use rgb_filled ("the goggle"), but torch is not installed. Try "pip install torch torchvision".')

            self.comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
            self.comp = torch.nn.DataParallel(self.comp).cuda()
            self.comp.load_state_dict(
                torch.load(os.path.join(gibson2.assets_path, 'networks', 'model.pth')))
            self.comp.eval()

        self.observation_space = gym.spaces.Dict(observation_space)

    def load_action_space(self):
        """
        Load action space
        """
        self.action_space = self.robots[0].action_space

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """
        self.current_step = 0
        self.collision_step = 0
        self.current_episode = 0
        self.collision_links = []

    def load(self):
        """
        Load environment
        """
        super(iGibsonEnv, self).load()
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def add_naive_noise_to_sensor(self, sensor_reading, noise_rate, noise_value=1.0):
        """
        Add naive sensor dropout to perceptual sensor, such as RGBD and LiDAR scan
        :param sensor_reading: raw sensor reading, range must be between [0.0, 1.0]
        :param noise_rate: how much noise to inject, 0.05 means 5% of the data will be replaced with noise_value
        :param noise_value: noise_value to overwrite raw sensor reading
        :return: sensor reading corrupted with noise
        """
        if noise_rate <= 0.0:
            return sensor_reading

        assert len(sensor_reading[(sensor_reading < 0.0) | (sensor_reading > 1.0)]) == 0,\
            'sensor reading has to be between [0.0, 1.0]'

        valid_mask = np.random.choice(2, sensor_reading.shape, p=[
                                      noise_rate, 1.0 - noise_rate])
        sensor_reading[valid_mask == 0] = noise_value
        return sensor_reading

    def get_depth(self):
        """
        :return: depth sensor reading, normalized to [0.0, 1.0]
        """
        depth = - \
            self.simulator.renderer.render_robot_cameras(modes=('3d'))[
                0][:, :, 2:3]
        # 0.0 is a special value for invalid entries
        depth[depth < self.depth_low] = 0.0
        depth[depth > self.depth_high] = 0.0

        # re-scale depth to [0.0, 1.0]
        depth /= self.depth_high
        depth = self.add_naive_noise_to_sensor(
            depth, self.depth_noise_rate, noise_value=0.0)

        return depth

    def get_rgb(self):
        """
        :return: RGB sensor reading, normalized to [0.0, 1.0]
        """
        return self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]

    def get_pc(self):
        """
        :return: pointcloud sensor reading
        """
        return self.simulator.renderer.render_robot_cameras(modes=('3d'))[0]

    def get_normal(self):
        """
        :return: surface normal reading
        """
        return self.simulator.renderer.render_robot_cameras(modes='normal')[0][:, :, :3]

    def get_seg(self):
        """
        :return: semantic segmentation mask, normalized to [0.0, 1.0]
        """
        seg = self.simulator.renderer.render_robot_cameras(modes='seg')[
            0][:, :, 0:1]
        if self.num_object_classes is not None:
            seg = np.clip(seg * 255.0 / self.num_object_classes, 0.0, 1.0)
        return seg

    def get_scan(self):
        """
        :return: LiDAR sensor reading, normalized to [0.0, 1.0]
        """
        laser_angular_half_range = self.laser_angular_range / 2.0
        if self.laser_link_name not in self.robots[0].parts:
            raise Exception('Trying to simulate LiDAR sensor, but laser_link_name cannot be found in the robot URDF file. Please add a link named laser_link_name at the intended laser pose. Feel free to check out assets/models/turtlebot/turtlebot.urdf and examples/configs/turtlebot_p2p_nav.yaml for examples.')
        laser_pose = self.robots[0].parts[self.laser_link_name].get_pose()
        angle = np.arange(-laser_angular_half_range / 180 * np.pi,
                          laser_angular_half_range / 180 * np.pi,
                          self.laser_angular_range / 180.0 * np.pi / self.n_horizontal_rays)
        unit_vector_local = np.array(
            [[np.cos(ang), np.sin(ang), 0.0] for ang in angle])
        transform_matrix = quat2mat(
            [laser_pose[6], laser_pose[3], laser_pose[4], laser_pose[5]])  # [x, y, z, w]
        unit_vector_world = transform_matrix.dot(unit_vector_local.T).T

        start_pose = np.tile(laser_pose[:3], (self.n_horizontal_rays, 1))
        start_pose += unit_vector_world * self.min_laser_dist
        end_pose = laser_pose[:3] + unit_vector_world * self.laser_linear_range
        results = p.rayTestBatch(start_pose, end_pose, 6)  # numThreads = 6

        # hit fraction = [0.0, 1.0] of self.laser_linear_range
        hit_fraction = np.array([item[2] for item in results])
        hit_fraction = self.add_naive_noise_to_sensor(
            hit_fraction, self.scan_noise_rate)
        scan = np.expand_dims(hit_fraction, 1)
        return scan

    def get_state(self, collision_links=[]):
        """
        :param collision_links: collisions from last time step
        :return: observation as a dictionary
        """
        state = OrderedDict()
        if 'task_obs' in self.output:
            state['task_obs'] = self.task.get_task_obs(self)
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
        if 'rgb_filled' in self.output:
            with torch.no_grad():
                tensor = transforms.ToTensor()(
                    (state['rgb'] * 255).astype(np.uint8)).cuda()
                rgb_filled = self.comp(tensor[None, :, :, :])[
                    0].permute(1, 2, 0).cpu().numpy()
                state['rgb_filled'] = rgb_filled
        if 'scan' in self.output:
            state['scan'] = self.get_scan()

        return state

    def run_simulation(self):
        """
        Run simulation for one action timestep (same as one render timestep in Simulator class)
        :return: collisions from this simulation
        """
        self.simulator_step()
        collision_links = list(p.getContactPoints(
            bodyA=self.robots[0].robot_ids[0]))
        return self.filter_collision_links(collision_links)

    def filter_collision_links(self, collision_links):
        """
        Filter out collisions that should be ignored
        :param collision_links: original collisions, a list of collisions
        :return: filtered collisions
        """
        new_collision_links = []
        for item in collision_links:
            # ignore collision with body b
            if item[2] in self.collision_ignore_body_b_ids:
                continue

            # ignore collision with robot link a
            if item[3] in self.collision_ignore_link_a_ids:
                continue

            # ignore self collision with robot link a (body b is also robot itself)
            if item[2] == self.robots[0].robot_ids[0] and item[4] in self.collision_ignore_link_a_ids:
                continue
            new_collision_links.append(item)
        return new_collision_links

    def populate_info(self, info):
        info['episode_length'] = self.current_step
        info['collision_step'] = self.collision_step

    def step(self, action):
        """
        apply robot's action and get state, reward, done and info, following OpenAI gym's convention
        :param action: a list of control signals
        :return: state, reward, done, info
        """
        self.current_step += 1
        if action is not None:
            self.robots[0].apply_action(action)
        collision_links = self.run_simulation()
        self.collision_links = collision_links
        self.collision_step += int(len(collision_links) > 0)

        state = self.get_state(collision_links)
        info = {}
        reward, info = self.task.get_reward(
            self, collision_links, action, info)
        done, info = self.task.get_termination(
            self, collision_links, action, info)
        self.task.step(self)
        self.populate_info(info)

        if done and self.automatic_reset:
            info['last_observation'] = state
            state = self.reset()
        return state, reward, done, info

    def check_collision(self, body_id):
        """
        :param body_id: pybullet body id
        :return: whether the given body_id has no collision
        """
        for _ in range(self.check_collision_loop):
            self.simulator_step()
            collisions = list(p.getContactPoints(bodyA=body_id))

            if logging.root.level <= logging.DEBUG:  # Only going into this if it is for logging --> efficiency
                for item in collisions:
                    logging.debug('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(
                        item[1], item[2], item[3], item[4]))

            if len(collisions) > 0:
                return False
        return True

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object
        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        obj.set_position_orientation([pos[0], pos[1], pos[2] + offset],
                                     quatToXYZW(euler2quat(*orn), 'wxyz'))

    def test_valid_position(self, obj_type, obj, pos, orn=None):
        """
        Test if the robot or the object can be placed with no collision
        :param obj_type: string "robot" or "obj"
        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :return: validity
        """
        assert obj_type in ['robot', 'obj']

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if obj_type == 'robot':
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if obj_type == 'robot' else obj.body_id
        has_collision = self.check_collision(body_id)
        return has_collision

    def land(self, obj_type, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation
        :param obj_type: string "robot" or "obj"
        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        assert obj_type in ['robot', 'obj']

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if obj_type == 'robot':
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if obj_type == 'robot' else obj.body_id

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if len(p.getContactPoints(bodyA=body_id)) > 0:
                land_success = True
                break

        if not land_success:
            print("WARNING: Failed to land")

        if obj_type == 'robot':
            obj.robot_specific_reset()
            obj.keep_still()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """
        self.current_episode += 1
        self.current_step = 0
        self.collision_step = 0
        self.collision_links = []

    def randomize_domain(self):
        if self.object_randomization_freq is not None:
            if self.current_episode % self.object_randomization_freq == 0:
                self.reload_model_object_randomization()
        if self.texture_randomization_freq is not None:
            if self.current_episode % self.texture_randomization_freq == 0:
                self.simulator.scene.randomize_texture()

    def reset(self):
        """
        Reset episode
        """
        self.randomize_domain()
        # move robot away from the scene
        self.robots[0].set_position([100.0, 100.0, 100.0])
        self.task.reset_scene(self)
        self.task.reset_agent(self)
        self.simulator.sync()
        state = self.get_state()
        self.reset_variables()

        return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    args = parser.parse_args()

    env = iGibsonEnv(config_file=args.config,
                     mode=args.mode,
                     action_timestep=1.0 / 10.0,
                     physics_timestep=1.0 / 40.0)

    step_time_list = []
    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        env.reset()
        for _ in range(100):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            print('reward', reward)
            if done:
                break
        print('Episode finished after {} timesteps, took {} seconds.'.format(
            env.current_step, time.time() - start))
    env.close()
