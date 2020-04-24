import os
import argparse
import time
import random
from IPython import embed
import cv2
import time
import collections
from collections import OrderedDict, namedtuple

import numpy as np
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from transforms3d.quaternions import quat2mat, qmult

import gym
import pybullet as p

import gibson2
from gibson2.core.physics.interactive_objects import InteractiveObj
from gibson2.utils.utils import rotate_vector_3d, l2_distance, quatToXYZW, parse_config, quatFromXYZW
from gibson2.envs.base_env import BaseEnv
from transforms3d.euler import euler2quat

# from core.viewer import CustomizedViewer
# from core.objects import VisualObject

# define navigation environments following Anderson, Peter, et al. 'On evaluation of embodied navigation agents.'
# arXiv preprint arXiv:1807.06757 (2018).
# https://arxiv.org/pdf/1807.06757.pdf

# debug_file = './infinite_loop.txt'
# debug_log = open(debug_file, 'w')


Episode = namedtuple('Episode',
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
	'collision',
	'timeout'
	])

class NavigateEnv(BaseEnv):
	def __init__(
			self,
			config_file,
			mode='headless',
			action_timestep=1 / 10.0,
			physics_timestep=1 / 240.0,
			automatic_reset=False,
			device_idx=0,
			verbose=False,
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
		
		# Metrics.
		self.n_steps = 0
		self.n_successes = 0
		self.n_collisions = 0
		self.n_timeouts = 0
		self.success = False
		self.distance_traveled = 0.0
		self.time_elapsed = 0.0
		self.episode_distance = 0.0
		self.spl_sum = 0 # shortest path length (SPL)
		self.spl = 0     # average shortest path length
		self.stored_episodes = collections.deque(maxlen=100)

		self.verbose = verbose
		self.robot_height = 0.1
		self.floor_num = None
		self.total_reward = 0

		# Camera
		self.camera_distance = 10.0
		self.camera_yaw = 60.0
		self.camera_pitch = -40.0
		self.robot_focus = None

		# Substitute gibson's viewer with our own.
		# Temporary solution to the bugs.
		if self.mode == 'gui':
			# self.simulator.viewer = None
			# self.simulator.viewer = CustomizedViewer(self.robots[0], renderer=self.simulator.renderer)
			# self.simulator.viewer.render = self.simulator.renderer
			# windows = set(self.config.get('sensor_inputs')).intersection({'rgb', 'depth'})
			# self.customized_viewer = CustomizedViewer(self.robots[0])
			self.customized_viewer = None
		else:
			self.customized_viewer = None

		if self.automatic_reset:
			# print('Automatic Reset!')
			self.reset()

	#########################################
	# Essential components of an environment.
	#########################################
	def load(self):
		super().load()
		# self.initial_pos = np.array(self.config.get('initial_pos', [0, 0, 0]))
		# self.initial_orn = np.array(self.config.get('initial_orn', [0, 0, 0]))

		# self.target_pos = np.array(self.config.get('target_pos', [5, 5, 0]))
		# self.target_orn = np.array(self.config.get('target_orn', [0, 0, 0]))

		# What is this additional_states_dim?
		self.obstacle_ids = []
		self.additional_states_dim = self.config.get('additional_states_dim', 0)
		self.auxiliary_sensor_dim = self.config.get('auxiliary_sensor_dim', 0)
		self.normalize_observation = self.config.get('normalize_observation', False)
		self.observation_normalizer = self.config.get('observation_normalizer', {})
		self.sensor_noise = self.config.get('sensor_noise', False)
		print('SENSOR_NOISE: {}'.format(self.sensor_noise))
		for key in self.observation_normalizer:
			self.observation_normalizer[key] = np.array(self.observation_normalizer[key])

		# termination condition
		self.stage = 0
		self.dist_tol = self.config.get('dist_tol', 0.2)
		self.max_step = self.config.get('max_step', float('inf'))
		self.gamma = self.config.get('gamma', 0.99)
		self.current_gamma = 1.0

		# reward
		self.reward_type = self.config.get('reward_type', 'dense')
		assert self.reward_type in ['dense', 'sparse', 'normalized_l2', 'l2', 'stage_sparse']

		self.success_reward = self.config.get('success_reward', 10.0)
		self.slack_reward = self.config.get('slack_reward', -0.01)
		self.death_z_thresh = self.config.get('death_z_thresh', 0.1)

		# reward weight
		self.potential_reward_weight = self.config.get('potential_reward_weight', 10.0)
		self.electricity_reward_weight = self.config.get('electricity_reward_weight', 0.0)
		self.stall_torque_reward_weight = self.config.get('stall_torque_reward_weight', 0.0)
		self.collision_reward_weight = self.config.get('collision_reward_weight', 0.0)
		# discount factor
		self.discount_factor = self.config.get('discount_factor', 1.0)

		# ignore the agent's collision with these body ids, typically ids of the ground and the robot itself.
		self.collision_ignore_body_b_ids = set(self.config.get('collision_ignore_body_b_ids', self.scene_ids))
		self.collision_ignore_link_a_ids = set(self.config.get('collision_ignore_link_a_ids', []))

		#
		self.load_core_parameters()
		self.load_customized_parameters()
		self.build_customized_scene()

		# Sensor inputs setting.
		# TODO: sensor: observations that are passed as network input, e.g. target position in local frame
		# TODO: auxiliary sensor: observations that are not passed as network input, but used to maintain the same
		# subgoals for the next T time steps, e.g. agent pose in global frame
		self.sensor_inputs = self.config.get('sensor_inputs')
		self.sensor_dim = self.additional_states_dim
		self.action_dim = self.robots[0].action_dim
		self.n_horizontal_rays = self.config.get('n_horizontal_rays', 128)
		self.n_vertical_beams = self.config.get('n_vertical_beams', 9)
		self.resolution = self.config.get('resolution', 64)

		observation_space = OrderedDict()
		self.initialize_core_sensors(observation_space)
		self.initialize_customized_sensors(observation_space)
		self.observation_space = gym.spaces.Dict(observation_space)

		self.action_space = self.robots[0].action_space

		self.visualize_initial = self.config.get('visualize_initial', False)
		self.visualize_target = self.config.get('visualize_target', False)

		self.visualize_initial_target()
		
		self.current_episode = 0    
	# Get potential only consideres the first two dimensions.


	def reset(self):
		# print('RESET! at time step: {}'.format(self.current_step)
		self.reset_agent()
		self.reset_other_objects()
		self.reset_core_variables()
		self.reset_customized_variables()
		# self.visualize_initial_target()
		if self.visualize_initial:
			self.initial_visual.set_position(self.initial_pos)
		if self.visualize_target:
			self.target_visual.set_position(self.current_target_position)
		state = self.get_state()
		return state

	
	def get_state(self, collision_links=[]):
		state = OrderedDict()
		self.get_core_states(state, collision_links)
		self.get_customized_states(state, collision_links)
		return state


	def get_reward(self, collision_links=[], action=None, info={}):
		reward = 0
		reward += self.get_core_reward(collision_links, action, info)
		reward += self.get_customized_reward(collision_links, action, info)
		return reward, info

	
	def get_termination(self, collision_links, info={}):
		done = False
		floor_height = 0.0 if self.floor_num is None else self.scene.get_floor_height(self.floor_num)
		if self.detect_collision(collision_links):
			terminate_reason = '***COLLISION***'
			# assert self.current_step > 0, "Collision shouldn't be penalized during resetting!"
			done = True
			info['success'] = False
			info['timeout'] = False
			info['collision'] = True
			self.n_collisions += 1
		elif self.reach_goal():
			terminate_reason = '***GOAL****'
			done = True
			info['success'] = True
			info['timeout'] = False
			info['collision'] = False
			self.n_successes += 1
		elif self.timeout():
			terminate_reason='***TIMEOUT***'
			done = True
			info['success'] = False
			info['timeout'] = True
			info['collision'] = False
			self.n_timeouts += 1
		elif self.robots[0].get_position()[2] > floor_height + self.death_z_thresh:
			terminate_reason='***DEATH***'
			done = True
			info['success'] = False
			info['timeout'] = False
			info['collision'] = False
		if done: # write summary.
			# print('Done! VERBOSE: {}'.format(self.verbose))

			# TODO: add more metrics (commented out the in the original code?)
			self.write_core_summary(info)
			self.write_customized_summary(info)
			episode = self.write_episodic_summary(info)
			self.stored_episodes.append(episode)
		return done, info


	def step(self, action):
		"""
		apply robot's action and get state, reward, done and info, following openAI gym's convention
		:param action: a list of control signals
		:return: state: state, reward, done, info
		"""
		info = {}
		self.current_step += 1
		# action = [0.1,0.1]
		# print('action: {}'.format(action))
		self.robots[0].apply_action(action)
		self.customized_step()
		cache = self.before_simulation()
		collision_links = self.run_simulation()
		self.after_simulation(cache, collision_links)
		state = self.get_state(collision_links)
		reward, info = self.get_reward(collision_links, action, info)
		done, info  = self.get_termination(collision_links, info)

		self.total_reward += reward * self.current_gamma
		self.current_gamma *= self.gamma
		
		if self.customized_viewer:
			self.customized_viewer.update(state)
			# self.customized_viewer.update()
		if done:
			if self.verbose:
				# print(info)
				keys = sorted(list(info.keys()))
				print("=" * 100)
				for key in keys:
					if key != 'last_observation':
						if key == 'collision_rate':
							print('{}: {}/{} = {}%'.format(key, self.n_collisions, self.current_episode, round(info[key], 2)))
						elif key == 'success_rate':
							print('{}: {}/{} = {}%'.format(key, self.n_successes, self.current_episode, round(info[key], 2)))
						elif key == 'timeout_rate':
							print('{}: {}/{} = {}%'.format(key, self.n_timeouts, self.current_episode, round(info[key], 2)))
						else:
							print('{}: {}'.format(key, info[key]))
			if self.automatic_reset:
				# print('Episode finished after {} timesteps'.format(self.current_step))
				info['last_observation'] = state
				state = self.reset()
		keyboard_events = p.getKeyboardEvents()
		if p.B3G_RETURN in keyboard_events:
			self.robot_focus = 0
		elif p.B3G_SHIFT in keyboard_events:
			# self.robot_focus = False
			p.resetDebugVisualizerCamera(30, 0.0, -89.0, \
				[0, 0, 0])
			self.robot_focus = None
		# if self.robot_focus:
		else:
			if self.robot_focus is not None:
				self.robot_focus += 1
		# elif p.B3G_RETURN in keyboard_events:
		if self.robot_focus == 1:
			self.robot_focus = 0
			robot_pos = self.robots[0].get_position()
			robot_XYZW = self.robots[0].get_orientation()
			quat = quatFromXYZW(robot_XYZW, 'xyzw')
			yaw = np.arctan2(2.0 * (quat[1] * quat[2] + quat[3] * quat[0]), \
				quat[3] ** 2 - quat[0] ** 2 - quat[1] ** 2 + quat[2] ** 2) + 45.0
			p.resetDebugVisualizerCamera(self.camera_distance, yaw, self.camera_pitch, \
				robot_pos[:3])
		return state, reward, done, info


	def customized_step(self):
		pass


	def run_simulation(self):
		collision_links = []
		for _ in range(self.simulator_loop):
			self.simulator_step()
			collision_links.append(list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0])))
		# TODO: See if can filter directly
		return self.filter_collision_links(collision_links)


	#########################################
	# Environment Initialization Helpers.
	#########################################
	def load_core_parameters(self):
		self.movements = self.config.get('movements')
		self.locations = self.config.get('locations')
		self.agent_sources = self.movements['agent']['sources']
		self.room_connectivity = self.config.get('connectivity')


	def load_customized_parameters(self):
		pass

	
	# Introduce customized components into the simulation.
	def build_customized_scene(self):
		pass

	def visualize_initial_target(self, cyl_length=1.2):
		if self.visualize_initial:
			self.initial_visual = VisualObject(visual_shape=p.GEOM_CYLINDER,
				rgba_color=[0, 0, 1, 0.4],
				radius=0.3,
				length=cyl_length,
				initial_offset=[0, 0, cyl_length / 2.0])
			self.initial_visual.load()
		if self.visualize_target:
			self.target_visual = VisualObject(visual_shape=p.GEOM_CYLINDER,
				rgba_color=[0, 1, 0, 0.4],
				radius=0.3,
				length=cyl_length,
				initial_offset=[0, 0, cyl_length / 2.0])
			if self.config.get('target_visible_to_agent', False):
				self.simulator.import_object(self.target_visual)
			else:
				self.target_visual.load()
	

	def initialize_core_sensors(self, observation_space):
		if 'sensor' in self.sensor_inputs:
			self.sensor_space = gym.spaces.Box(low=-np.inf,
				high=np.inf,
				shape=(self.sensor_dim,),
				dtype=np.float32)
			observation_space['sensor'] = self.sensor_space
		if 'auxiliary_sensor' in self.sensor_inputs:
			self.auxiliary_sensor_space = gym.spaces.Box(low=-np.inf,
				high=np.inf,
				shape=(self.auxiliary_sensor_dim,),
				dtype=np.float32)
			observation_space['auxiliary_sensor'] = self.auxiliary_sensor_space
		if 'pointgoal' in self.sensor_inputs:
			self.pointgoal_space = gym.spaces.Box(low=-np.inf,
				high=np.inf,
				shape=(2,),
				dtype=np.float32)
			observation_space['pointgoal'] = self.pointgoal_space
		if 'rgb' in self.sensor_inputs:
			self.rgb_space = gym.spaces.Box(low=-np.inf,
				high=np.inf,
				shape=(self.resolution, self.resolution, 3),
				dtype=np.float32)
			observation_space['rgb'] = self.rgb_space
		if 'depth' in self.sensor_inputs:
			self.depth_space = gym.spaces.Box(low=-np.inf,
				high=np.inf,
				shape=(self.resolution, self.resolution, 1),
				dtype=np.float32)
			observation_space['depth'] = self.depth_space
		if 'scan' in self.sensor_inputs:
			self.scan_mode = self.config.get('scan_mode', 'xyz')
			scan_channels = 1 if self.scan_mode == 'dist' else 3
			self.scan_space = gym.spaces.Box(low=-np.inf, 
				high=np.inf,
				shape=(self.n_horizontal_rays * self.n_vertical_beams, scan_channels),
				dtype=np.float32)
			observation_space['scan'] = self.scan_space
		if 'rgb_filled' in self.sensor_inputs:  # use filler
			self.comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
			self.comp = torch.nn.DataParallel(self.comp).cuda()
			self.comp.load_state_dict(
				torch.load(os.path.join(gibson2.assets_path, 'networks', 'model.pth')))
			self.comp.eval()
		if 'waypoints' in self.sensor_inputs:
			self.waypoints_space = gym.spaces.Box(low=-np.inf, high=np.inf,
			shape=(self.config['waypoints']*2,),  # waypoints * len([x_pos, y_pos])
			dtype=np.float32)
			observation_space['waypoints'] = self.waypoints_space


	def initialize_customized_sensors(self, observation_space):
		pass
	
	def get_auxiliary_sensor(self, collision_links=[]):
		return np.array([])
	

	#########################################
	# Reset Helpers.
	#########################################
	def reset_core_variables(self):
		self.current_episode += 1
		self.current_step = 0
		self.potential = self.get_potential()
		self.initial_potential = self.potential
		self.normalized_potential = 1.0
		self.kinematic_disturbance = 0.0
		self.dynamic_disturbance_b = 0.0
		self.dynamic_disturbance_a = 0.0
		self.path_length = 0.0
		self.total_reward = 0.0
		self.current_gamma = 1.0
		self.agent_trajectory = []

	
	def reset_agent(self):
		self.robots[0].robot_specific_reset()

		# Check if robot has reached target in previous episodes. 
		# robot_position = self.robots[0].get_position()
		# initial, target = random.choice(self.agent_initial_target_list)

		# Reset robot's initial position.
		reset_complete = False
		while not reset_complete:
			# print("Resetting robot state!!!!")
			initial = random.choice(self.agent_sources)
			initial_x = self.locations[initial]['center_x']
			initial_y = self.locations[initial]['center_y']
			x_range_radius = self.locations[initial]['x_range_radius']
			y_range_radius = self.locations[initial]['y_range_radius']
			self.initial_pos = np.array([*self._get_random_point(initial_x, initial_y, 
				x_range_radius, y_range_radius), self.robot_height])
			self.robots[0].set_position(pos=self.initial_pos)
			self.robots[0].set_orientation(
				orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz'))
			# Check for collision.
			collision_links = self.run_simulation()
			# time.sleep(1)
			reset_complete = self._no_collision(collision_links)

		# Reset robot's target position.
		#TODO: check if target collides with a static obstacle?
		target = random.choice(self.room_connectivity[initial])
		target_x = self.locations[target]['center_x']
		target_y = self.locations[target]['center_y']
		x_range_radius = self.locations[target]['x_range_radius']
		y_range_radius = self.locations[target]['y_range_radius']
		reset_complete = False
		while not reset_complete:
			# print('Resetting robot target!!!')
			self.current_target_position = np.array([*self._get_random_point(target_x, target_y,
				x_range_radius, y_range_radius), 0.0])
			dist = l2_distance(self.initial_pos[:2], self.current_target_position[:2])
			reset_complete = dist >= 1.0
		# Semi-hard code agent's current location.
		if initial.startswith('h'):
			self.agent_location = 'hallway'
		elif initial.startswith('c'):
			self.agent_location = 'crossing'
		elif initial.startswith('d'):
			self.agent_location = 'doorway'


	def reset_customized_variables(self):
		pass

			
	def reset_other_objects(self):
		pass

	def get_agent_location(self):
		return self.agent_location


	#########################################
	# State Getters.
	#########################################
	def get_core_states(self, state, collision_links=[]):
		# Omit 'normal' and 'seg' 'depth_seg', for now because we are not using them.
		if 'sensor' in self.sensor_inputs:
			state['sensor'] = self.get_additional_states()
			# print('target: {}'.format(state['sensor']))
		if 'auxiliary_sensor' in self.sensor_inputs:
			state['auxiliary_sensor'] = auxiliary_sensor
		if 'pointgoal' in self.sensor_inputs:
			state['pointgoal'] = self.get_additional_states()[:2]
		if 'rgb' in self.sensor_inputs:
			state['rgb'] = self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]
		if 'depth' in self.sensor_inputs:
			depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
			state['depth'] = depth
		if 'rgb_filled' in self.sensor_inputs:
			with torch.no_grad():
				tensor = transforms.ToTensor()((state['rgb'] * 255).astype(np.uint8)).cuda()
				rgb_filled = self.comp(tensor[None, :, :, :])[0].permute(1, 2, 0).cpu().numpy()
				state['rgb_filled'] = rgb_filled
		if 'scan' in self.sensor_inputs:
			assert 'scan_link' in self.robots[0].parts, "Requested scan but no scan_link"
			state['scan'] = self._get_lidar_state()
		if 'waypoints' in self.sensor_inputs:
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

	
	def get_customized_states(self, state, collision_links=[]):
		pass
		

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

	
	def _get_lidar_state(self):
		pose_camera = self.robots[0].parts['scan_link'].get_pose()
		# The maximum lidar range may not always be 360 degree?
		angle = np.arange(0, 2 * np.pi, 2 * np.pi / float(self.n_horizontal_rays))
		elev_bottom_angle = -15. * np.pi / 180.
		elev_top_angle = 15. * np.pi / 180
		if self.n_vertical_beams == 1:
			elev_angle = [0.0]
		else:
			elev_angle = np.linspace(elev_bottom_angle, elev_top_angle, num=self.n_vertical_beams)
			# I feel the original implementation is still buggy. 
			# elev_angle = np.arange(elev_bottom_angle, elev_top_angle, \
				# (elev_top_angle - elev_bottom_angle) / float(self.n_vertical_beams-1))
		orig_offset = np.vstack([
				np.vstack([np.cos(angle),
				np.sin(angle),
				np.repeat(np.tan(elev_ang), angle.shape)]).T for elev_ang in elev_angle
			])

		# print('orig_offset: {}'.format(orig_offset.shape))
		transform_matrix = quat2mat([pose_camera[-1], pose_camera[3], pose_camera[4], pose_camera[5]])
		# print('shape of transform matrix: {}'.format(transform_matrix))
		offset = orig_offset.dot(np.linalg.inv(transform_matrix))
		pose_camera = pose_camera[None, :3].repeat(self.n_horizontal_rays * self.n_vertical_beams, axis=0)
		results = p.rayTestBatch(pose_camera, pose_camera + offset * 30)
		# print('results: {}'.format(results[0]))
		hit = np.array([item[0] for item in results])
		dist = np.array([item[2] for item in results])
		valid_pts = (dist < 1. - 1e-5) & (dist > 0.1 / 30) & (hit != self.robots[0].robot_ids[0]) & (hit != -1)
		if self.scan_mode == 'dist':
			dist[~valid_pts] = 1.0
			dist *= 30
			return dist
		if self.scan_mode == 'xyz':
			dist[~valid_pts] = 1.0
			dist *= 30
			lidar_scan = np.expand_dims(dist, 1) * orig_offset
			if self.sensor_noise:
				noise = np.random.normal(size=lidar_scan.shape)
				lidar_scan += noise
			return lidar_scan
	

	#########################################
	# Reward Getters.
	#########################################
	def get_core_reward(self, collision_links=[], action=None, info={}):
		# slack reward: penalty for each timestep.
		reward = self.slack_reward
		# print('after slack reward: {}'.format(reward))
		
		# potential reward: reward closeness to the target.
		self.potential = self.get_potential()
		# print('new potential: {}'.format(new_potential))
		if self.potential_reward_weight > 0:
			new_normalized_potential = self.potential / self.initial_potential
			potential_reward = self.normalized_potential - new_normalized_potential
			reward += potential_reward * self.potential_reward_weight
			self.normalized_potential = new_normalized_potential
			# reward += (new_potential - self.potential) * self.potential_reward_weight
			# self.potential = new_potential
		# print('after potential reward: {}'.format(reward))

		# electricity reward:
		if self.electricity_reward_weight > 0:
			electricity_reward = np.abs(self.robots[0].joint_speeds * self.robots[0].joint_torque).mean().item()
			reward += electricity_reward * self.electricity_reward_weight

		# stall torque reward:
		if self.stall_torque_reward_weight > 0:
			stall_torque_reward = np.square(self.robots[0].joint_torque).mean()
			reward += stall_torque_reward * self.stall_torque_reward_weight

		# collision reward.
		self.collision = self.detect_collision_for_reward(collision_links)
		# info['collision_reward'] = int(self.collision) * self.collision_reward_weight
		# reward += info['collision_reward']
		reward += int(self.collision) * self.collision_reward_weight
		# print('after collision reward: {}'.format(reward))
		# self.collision_step += int(self.collision) # What is this?

		# goal reward.
		if self.potential < self.dist_tol:
			reward += self.success_reward
		# print('after success: {}'.format(reward))

		return reward


	def get_customized_reward(self, collision_links=[], action=None, info={}):
		return 0


	#########################################
	# Termination Helpers.
	#########################################
	# Can have different definitions.
	# Do the easy version for now. Since the other version tends to raise problems.

	def detect_collision_for_reward(self, collision_links):
		return not self._no_collision(collision_links)

	def detect_collision(self, collision_links):
		return not self._no_collision(collision_links)

	# def detect_collision(self, collision_links):
		# robot_velocity = self.robots[0].get_velocity()
		# return not self._no_collision(collision_links) and np.linalg.norm(robot_velocity[:2]) > 0.05
	
	
	
	def reach_goal(self):
		return self.potential < self.dist_tol

	
	def timeout(self):
		return self.current_step >= self.max_step


	#########################################
	# Miscellaneous Helpers.
	#########################################
	def write_core_summary(self, info):
		info['episodes'] = self.current_episode
		info['episode_length'] = self.current_step
		info['success_rate'] = 0 if self.current_episode == 0 else 100 * self.n_successes / (self.current_episode)
		info['collision_rate'] = 0 if self.current_episode == 0 else 100 * self.n_collisions / (self.current_episode)
		info['timeout_rate'] = 0 if self.current_episode == 0 else 100 * self.n_timeouts / (self.current_episode)
		info['path_length'] = 0
		info['return'] = self.total_reward
		info['shortest_path_length'] = None if self.current_episode == 0 else [self.spl]

	
	def write_customized_summary(self, info):
		pass


	def filter_collision_links(self, collision_links):
		filtered_collision_links = [[item for item in collision_per_sim_step
			if item[2] not in self.collision_ignore_body_b_ids and 
			item[3] not in self.collision_ignore_link_a_ids]
			for collision_per_sim_step in collision_links]
		return filtered_collision_links

	
	# Get robot's current pose.
	def get_position_of_interest(self):
		if self.config['task'] == 'pointgoal':
			return self.robots[0].get_position()
		elif self.config['task'] == 'reaching':
			return self.robots[0].get_end_effector_position()

	
	# Get potential only consideres the first two dimensions.
	def get_potential(self):
		return l2_distance(self.current_target_position[:2], self.get_position_of_interest()[:2])


	def _no_collision(self, collision_links):
		if isinstance(collision_links, list):
			return all(map(self._no_collision, collision_links))
		return False


	def get_stored_episodes(self):
		return self.stored_episodes


	def before_simulation(self):
		return None


	def after_simulation(self, cache, collision_links):
		return


	def write_episodic_summary(self, info):
		episode = Episode(
				env=None,
				agent=self.robots[0].model_file,
				initial_pos=self.initial_pos,
				target_pos=self.current_target_position,
				success=float(info['success']),
				collision=float(info['collision']),
				timeout=float(info['timeout']),
				geodesic_distance=0,
				shortest_path=0,
				agent_trajectory=None,
				object_files=None,
				object_trajectory=None,
				path_efficiency=0,
				kinematic_disturbance=0,
				dynamic_disturbance_a=0,
				dynamic_disturbance_b=0,
				collision_step=0
				)
		return episode


	# Get a random 2D position given center (x, y) and randomization range (range_x, range_y) 
	def _get_random_point(self, x, y, range_x=0, range_y=0):
		deviation = np.random.random_sample(2) * 2 - 1
		center = np.array([x, y])
		deviation = np.array([range_x, range_y]) * deviation
		return center + deviation
