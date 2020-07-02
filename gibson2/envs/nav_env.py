import os
import argparse
import time
import random
from IPython import embed
import cv2
import json
import yaml
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
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import VisualObject, InstanceGroup, MeshRenderer
from transforms3d.euler import euler2quat
import matplotlib.pyplot as plt

from gibson2.core.viewer import CustomizedViewer
from gibson2.core.objects import VisualObject
from gibson2.utils.global_planner import GlobalPlanner

from gym.wrappers.monitoring.video_recorder import ImageEncoder
from datetime import datetime

from scipy.spatial.transform import Rotation

from statistics import mean, variance

from gibson2.core.pedestrians.state import ObservableState
from gibson2.core.pedestrians.action import ActionXY, ActionRot

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
			global_planner=None,
			record=False,
			**kwargs
	):
		super(NavigateEnv, self).__init__(config_file=config_file, 
			mode=mode, 
	 		device_idx=device_idx)

		self.config_file = config_file
		self.automatic_reset = automatic_reset

		# Global planner
		if global_planner:
			map_path = kwargs.pop('map_path', None)
			trav_map_resolution = kwargs.pop('trav_map_resolution', 0.1)
			trav_map_erosion = kwargs.pop('trav_map_erosion', 0.8) # robot footprint radius.
			waypoint_resolution = kwargs.pop('waypoint_resolution', 0.5)
			planner_algorithm = kwargs.pop('planner_algorithm', 'astar')
			graph_path = kwargs.pop('graph_path', 'connectivity.p')
			trav_map_path = kwargs.pop('trav_map_path', 'trav_map.png')
			self.global_planner = GlobalPlanner(map_path, trav_map_resolution,
				trav_map_erosion, waypoint_resolution, planner_algorithm)
			self.global_planner.build_connectivity_graph(graph_path, trav_map_path)
		else:
			self.global_planner = None

		# simulation
		self.mode = mode
		self.action_timestep = action_timestep
		self.physics_timestep = physics_timestep
		self.simulator.set_timestep(physics_timestep)
		self.simulator_loop = int(self.action_timestep / self.simulator.timestep)

		self.current_step = 0
		self.global_step = 0

		# Metrics.
		self.n_steps = 0
		self.n_successes = 0
		self.n_collisions = 0
		self.n_timeouts = 0
		self.success = False
		self.distance_traveled = 0.0
		self.time_elapsed = 0.0
		self.last_vel_linear = 0.0
		self.last_vel_angular = 0.0
		self.last_accel_linear = 0.0
		self.last_accel_angular = 0.0
		self.total_jerk	= 0.0
		self.episode_distance = 0.0
		self.spl_sum = 0 # shortest path length (SPL)
		self.spl_mean = 0     # average shortest path length
		self.stored_episodes = collections.deque(maxlen=100)
		self.distance_to_nearest_pedestrian = np.inf
		self.log_distance_to_nearest_pedestrian = list()

		self.verbose = verbose
		self.robot_height = 0.1
		self.floor_num = None
		self.total_reward = 0
		self.robot_trajectories = []

		# Camera
		self.camera_distance = 10.0
		self.camera_yaw = 0.0
		self.camera_pitch = -89.0
		self.robot_focus = None
		
		self.max_v_a = 0.0
		self.min_v_a = 1000

		# record
		self.record = record
		self.metadata['render.modes'] = ['rgb_array']
		self.recording = False
		if self.record:
			self.video_length = kwargs.pop('video_length', 4000)
			self.video_episode_length = kwargs.pop('video_episode_length', 2000)
			self.video_folder = kwargs.pop('video_folder', 'video')
			self.frames_per_sec = kwargs.pop('frames_per_sec', 30)
			self.output_frames_per_sec = kwargs.pop('output_frames_per_sec', 30)
			self.start_record()

		self.keep_trajectory = kwargs.pop('keep_trajectory', False)
		# self.keep_trajectory = True
		if self.keep_trajectory:
			self.history = dict()
			self.history_path = os.path.join(self.root_dir, 'history')
			if not os.path.exists(self.history_path):
				os.makedirs(self.history_path)
			self.history_json = os.path.join(self.history_path, 'history-{}.json'.format(str(datetime.now())))
		
		self.build_scene()
		self.visualize_initial_target()
		if self.mode == 'gui':
			self.simulator.viewer = None
			# self.simulator.viewer = CustomizedViewer(self.robots[0], renderer=self.simulator.renderer)
			# self.simulator.viewer.render = self.simulator.renderer
			# windows = set(self.config.get('sensor_inputs')).intersection({'rgb', 'depth'})
			# self.customized_viewer = CustomizedViewer(self.robots[0])
			self.customized_viewer = None
		else:
			self.customized_viewer = None

		if self.automatic_reset:
			self.reset()


	#########################################
	# Essential components of an environment.
	#########################################
	def load(self):
		super(NavigateEnv, self).load()

		# Robot name
		robot_mode = self.config.get('robot').lower()
		if 'turtlebot' in robot_mode:
			self.robot_name = 'turtlebot'
		elif 'jr2' in robot_mode:
			self.robot_name = 'jr2'
		else:
			raise "robot is not defined."

		# termination condition
		self.root_dir = self.config.get('root_dir', 'test')
		self.stage = 0
		self.dist_tol = self.config.get('dist_tol', 0.5)
		self.max_step = self.config.get('max_step', 500)
		self.gamma = self.config.get('gamma', 0.99)
		self.current_gamma = 1.0

		# Global planner
		replan_frequency = self.config.get('planner_frequency', 1)
		self.replan_interval = self.max_step // replan_frequency
		# self.replan_interval = 1

		# reward
		self.reward_type = self.config.get('reward_type', 'geodesic')
		assert self.reward_type in ['dense', 'sparse', 'geodesic', 'normalized_l2', 'l2', 'stage_sparse']

		self.success_reward = self.config.get('success_reward')
		self.slack_reward = self.config.get('slack_reward')
		self.death_z_thresh = self.config.get('death_z_thresh', 0.1)

		# reward weight
		self.potential_reward_weight = self.config.get('potential_reward_weight', 1.0)
		self.electricity_reward_weight = self.config.get('electricity_reward_weight', 0.0)
		self.stall_torque_reward_weight = self.config.get('stall_torque_reward_weight', 0.0)
		self.collision_reward_weight = self.config.get('collision_reward_weight', -1.0)

		# ignore the agent's collision with these body ids, typically ids of the ground and the robot itself.
		self.collision_ignore_body_b_ids = set(self.config.get('collision_ignore_body_b_ids', [0, 1, 2]))
		self.collision_ignore_link_a_ids = set(self.config.get('collision_ignore_link_a_ids', []))

		self.obstacle_ids = []
		self.load_parameters()
		# self.build_scene()

		# Sensor inputs setting.
		# TODO: sensor: observations that are passed as network input, e.g. target position in local frame
		# TODO: auxiliary sensor: observations that are not passed as network input, but used to maintain the same
		# subgoals for the next T time steps, e.g. agent pose in global frame
		# Sensor input
		self.additional_states_dim = self.config.get('additional_states_dim')
		self.auxiliary_sensor_dim = self.config.get('auxiliary_sensor_dim')
		self.observation_normalizer = self.config.get('observation_normalizer', {})
		self.sensor_noise = self.config.get('sensor_noise')
		for key in self.observation_normalizer:
			self.observation_normalizer[key] = np.array(self.observation_normalizer[key])
		self.sensor_inputs = self.config.get('sensor_inputs')
		self.sensor_dim = self.additional_states_dim
		self.action_dim = self.robots[0].action_dim
		
		self.n_horizontal_rays = self.config.get('n_horizontal_rays')
		self.n_vertical_beams = self.config.get('n_vertical_beams')
		self.resolution = self.config.get('resolution')
		self.num_waypoints = self.config.get('waypoints')
		# The robot is considered stopped below this speed for pedestrian collision purposes
		self.zero_velocity_threshold = self.config.get('zero_velocity_threshold', 0.01)
		
		observation_space = OrderedDict()
		self.initialize_sensors(observation_space)
		
		self.observation_space = gym.spaces.Dict(observation_space)

		self.action_space = self.robots[0].action_space

		self.visualize_initial = self.config.get('visualize_initial')
		self.visualize_target = self.config.get('visualize_target')
		self.visualize_waypoints = self.config.get('visualize_waypoints')
		self.visualize_lidar = self.config.get('visualize_lidar', False)
		
		self.max_linear_velocity = self.config.get("max_linear_velocity", 0.3)
		self.max_angular_velocity = self.config.get("max_angular_velocity", 1.0)
						
		self.current_episode = 0


	def reset(self):
		if self.keep_trajectory:
			self.trajectory = {'linear_vel': [], 'linear_action': [],
			'angular_vel': [], 'angular_action': [], 'pos': [], 
			'instant_linear_vel': [], 'instant_angular_vel': [], 'instant_pos': []}
		self.reset_agent()
		self.reset_other_objects()
		self.reset_variables()
		if self.visualize_initial or self.record:
			self.initial_visual.set_position(self.initial_pos)
		if self.visualize_target:
			self.target_visual.set_position(self.current_target_position)
		state = self.get_state()
		return state

	
	def get_state(self, collision_links=[]):
		state = OrderedDict()
		self.get_sensor_states(state, collision_links)
		return state


	def get_reward(self, collision_links=[], action=None, info={}):
		reward_core, info = self.get_core_reward(collision_links, action, info)
		reward_custom = self.get_customized_reward(collision_links, action, info)
		reward = reward_core + reward_custom
		return reward, info
	
	def get_termination(self, collision_links, info={}):
		done = False
		floor_height = 0.0 if self.floor_num is None else self.scene.get_floor_height(self.floor_num)
		if self.detect_collision(collision_links):
			terminate_reason = '***COLLISION***'
			done = True
			info['success'] = False
			info['timeout'] = False
			info['collision'] = True
			self.n_collisions += 1
			
			print("COLLISION!")
# 			# Was this a pedestrian collision?
# 			for collision in collision_links[1]:
# 				if collision[2] in self.pedestrian_gibson_ids:
# 					self.n_pedestrian_collisions += 1
# 					rob_vel = np.linalg.norm(np.array(self.robots[0].get_velocity())[:2])
# 					if rob_vel > self.zero_velocity_threshold:
# 						self.n_robot_hit_pedestrian += 1
# 					else:
# 						self.n_pedestrian_hit_robot += 1
# 					print("PEDESTRIAN COLLISION AT STEP: ", self.current_step)
			
		elif self.reach_goal():
			terminate_reason = '***GOAL****'
			done = True
			info['success'] = True
			info['timeout'] = False
			info['collision'] = False
			self.n_successes += 1
			self.time_elapsed += self.current_step * self.action_timestep
			print("SUCCESS AT STEP: ", self.current_step)
		elif self.timeout():
			terminate_reason='***TIMEOUT***'
			done = True
			info['success'] = False
			info['timeout'] = True
			info['collision'] = False
			self.n_timeouts += 1
			print("TIMEOUT AT STEP: ", self.current_step)

# 		elif self.robots[0].get_position()[2] > floor_height + self.death_z_thresh:
# 			terminate_reason='***DEATH***'
# 			done = True
# 			info['success'] = False
# 			info['timeout'] = False
# 			info['collision'] = False
		if done: # write summary.
			# TODO: add more metrics (commented out the in the original code?)
			self.log_distance_to_nearest_pedestrian.append(self.distance_to_nearest_pedestrian)
			
			# shortest path length (distance_traveled / shortest_distance weighted by success (1 or 0))
			if info['success']:
				self.spl_sum += self.episode_distance / self.shortest_distance
			self.spl_mean = self.spl_sum / self.current_episode
			
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
		
		old_position = self.robots[0].get_position()

		self.robots[0].apply_action(action)
		
# 		lin_vel = np.linalg.norm(self.robots[0].get_velocity())
# 		ang_vel = np.linalg.norm(self.robots[0].get_angular_velocity())
# 		
# 		if ang_vel > 0.1 and ang_vel < self.min_v_a:
# 			self.min_v_a = ang_vel
# 			print("MIN:", ang_vel)
# 
# 		if ang_vel > self.max_v_a:
# 			self.max_v_a = ang_vel
# 			print("MAX:", ang_vel)
		
# 		print(action)
# 		
# 		linear_vel = action[0] * self.max_linear_velocity
# 		
# 		if linear_vel < 0.2:
# 			linear_vel = 0
# 			action[0] = action[0] - self.max_linear_velocity
		
# 		# Move the robot using ORCA
# 		ob = []
# 		location = 'hallway'
# 		for ped_pid in self.pedestrians[location]:
# 			ped = self.pedestrians[location][ped_pid]['object']
# 			ob.append(ped.get_observable_state())
# 			
# 		if self.walls:
# 			walls_config = list(zip(self.walls['walls_pos'], self.walls['walls_dim']))
# 		else:
# 			walls_config = list()
# 
# 		#action = ped.act(ob, walls=walls_config, obstacles=[])
# 		action = ActionRot(v=-1.0, r=0.1)
# 		#print(action)
# 		self.robots[0].apply_action(action)
				
		self.customized_step()
		cache = self.before_simulation() #TODO: use this function for waypoints.
		collision_links = self.run_simulation(keep_trajectory=self.keep_trajectory)
		self.after_simulation(cache, collision_links)
		state = self.get_state(collision_links)
		reward, info = self.get_reward(collision_links, action, info)
		done, info  = self.get_termination(collision_links, info)

		new_position = self.robots[0].get_position()
		distance_traveled = np.linalg.norm(new_position - old_position)
		self.episode_distance += distance_traveled

		linear_velocity = (action[0] + self.max_linear_velocity) / 2.0 # m/s
		angular_velocity = action[1] # rad/s

		delta_vel_linear = self.last_vel_linear - linear_velocity
		accel_linear = delta_vel_linear / self.action_timestep

		delta_vel_angular = self.last_vel_angular - angular_velocity
		accel_angular = delta_vel_angular / self.action_timestep

		jerk_linear = abs((self.last_accel_linear - accel_linear) / self.action_timestep)
		jerk_angular = abs((self.last_accel_angular - accel_angular) / self.action_timestep)

		self.total_jerk += (jerk_linear + jerk_angular)

		self.last_vel_linear = linear_velocity
		self.last_vel_angular = angular_velocity
		self.last_accel_linear = accel_linear
		self.last_accel_angular = accel_angular

		nearest_pedestrian = self._distance_to_closest_pedestrian()
		
		if nearest_pedestrian < self.distance_to_nearest_pedestrian:
			self.distance_to_nearest_pedestrian = nearest_pedestrian

		# vel = p.getBaseVelocity(22)
		# print('another new velocity: {}'.format(vel))

		if self.keep_trajectory:
			linear_action, angular_action = float(action[0]), float(action[1])
			self.trajectory['linear_action'].append(linear_action)
			self.trajectory['angular_action'].append(angular_action)

			# vel = p.getBaseVelocity(self.robots[0].robot_ids[0])
			# linear_vel = np.array(vel[0])
			# angular_vel = np.array(vel[1])
			# linear_vel = float(np.sqrt(linear_vel[0] ** 2 + linear_vel[1] ** 2))
			# angular_vel = float(angular_vel[2])
			# self.trajectory['linear_vel'].append(linear_vel)
			# self.trajectory['angular_vel'].append(angular_vel)
			# self.trajectory['pos'].append(self.robots[0].get_position())

		self.total_reward += reward * self.current_gamma
		self.current_gamma *= self.gamma

		if self.customized_viewer:
			self.customized_viewer.update(state)

		if self.record:
			if self._video_start() and not self.recording:
				self.recording = True
			if self._video_end() and self.recording:
				self.recording = False
				self.close_video_recorder()

			if self.recording:
				if self._new_video_episode():
					self.start_video_recorder()
				self.capture_frame()
				self.step_id += 1


		if done:
			if self.keep_trajectory:
				self.history[self.current_episode] = self.trajectory.copy()
				if self.current_episode % 1 == 0:
					with open(self.history_json, 'w') as jf:
						json.dump(self.history, jf)
			if self.verbose:
				keys = sorted(list(info.keys()))
				print("=" * 100)
				for key in keys:
					if key != 'last_observation':
						if key == 'non_pedestrian_collision_rate':
							print('{}: {}/{} = {}%'.format(key, (self.n_collisions - self.n_pedestrian_collisions), self.current_episode, round(info[key], 2)))
						elif key == 'pedestrian_collision_rate':
							print('{}: {}/{} = {}%'.format(key, self.n_pedestrian_collisions, self.current_episode, round(info[key], 2)))
						elif key == 'robot_hit_pedestrian_rate':
							print('{}: {}/{} = {}%'.format(key, self.n_robot_hit_pedestrian, self.current_episode, round(info[key], 2)))
						elif key == 'pedestrian_hit_robot_rate':
							print('{}: {}/{} = {}%'.format(key, self.n_pedestrian_hit_robot, self.current_episode, round(info[key], 2)))
						elif key == 'success_rate':
							print('{}: {}/{} = {}%'.format(key, self.n_successes, self.current_episode, round(info[key], 2)))
						elif key == 'timeout_rate':
							print('{}: {}/{} = {}%'.format(key, self.n_timeouts, self.current_episode, round(info[key], 2)))
						elif key == 'mean_nearest_pedestrian_distance':
							print('{}: {}'.format(key, round(info[key], 2)))
						elif key == 'variance_nearest_pedestrian_distance':
							print('{}: {}'.format(key, round(info[key], 2)))
						elif key == 'shortest_path_length':
							print('{}: {}'.format(key, round(self.spl_mean, 2)))
						elif key == 'time_elapsed':
							if  self.n_successes > 0:
								print('{}: {}'.format(key, round(self.time_elapsed / self.n_successes, 2)))
							else:
								print('{}: {}'.format(key, 0.0))
						elif key == 'ave_jerk':
							print('{}: {}'.format(key, round(self.total_jerk / self.current_episode, 2)))
						else:
							print('{}: {}'.format(key, info[key]))
			if self.automatic_reset:
				info['last_observation'] = state
				state = self.reset()


		keyboard_events = p.getKeyboardEvents()
		
		# Camera focus on the robot.
		if p.B3G_RETURN in keyboard_events:
			self.robot_focus = 0
		# Top down view.
		elif p.B3G_SHIFT in keyboard_events:
			p.resetDebugVisualizerCamera(10, 0.0, -89.0, \
				self.robots[0].get_position()[:3])
			self.robot_focus = None
		
		focus_interval = 1
		if self.robot_focus is not None:
			if self.robot_focus % focus_interval == 0:
				self.camera_target = self.robots[0].get_position()
				p.resetDebugVisualizerCamera(10, self.camera_yaw, self.camera_pitch, \
					self.camera_target)
			self.robot_focus += 1

		return state, reward, done, info

	def customized_step(self):
		pass

	def run_simulation(self, keep_trajectory=False):
		collision_links = []
		for _ in range(self.simulator_loop):
			self.simulator_step()
			self.global_step += 1
			self.global_time += self.physics_timestep
			collision_links.append(list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0])))

			if keep_trajectory:
				vel = p.getBaseVelocity(self.robots[0].robot_ids[0])
				linear_vel = np.array(vel[0])
				angular_vel = np.array(vel[1])
				linear_vel = float(np.sqrt(linear_vel[0] ** 2 + linear_vel[1] ** 2))
				angular_vel = float(angular_vel[2])
				rob_pos = self.robots[0].get_position()
				rob_pos = [float(p) for p in rob_pos]
				self.trajectory['instant_linear_vel'].append(linear_vel)
				self.trajectory['instant_angular_vel'].append(angular_vel)
				self.trajectory['instant_pos'].append(rob_pos)

		# TODO: See if can filter directly
		
		return self.filter_collision_links(collision_links)

	#########################################
	# Environment Initialization Helpers.
	#########################################
	def load_parameters(self):
		self.movements = self.config.get('movements')
		self.locations = self.config.get('locations')
		try:
			self.movements['agent']['sources']
		except:
			self.movements['agent']['sources'] = ['h1-rob', 'h2-rob']
		self.agent_sources = self.movements['agent']['sources']
		self.room_connectivity = self.config.get('connectivity')


	def load_customized_parameters(self):
		pass

	# Introduce customized components into the simulation.
	def build_scene(self):
		pass

	def visualize_initial_target(self, cyl_length=1.2):
		if self.visualize_initial or self.record:
			self.initial_visual = VisualObject(visual_shape=p.GEOM_CYLINDER,
				rgba_color=[0, 0, 1, 0.4],
				radius=0.3,
				length=cyl_length,
				initial_offset=[0, 0, cyl_length / 2.0])
			if True:
				self.simulator.import_object(self.initial_visual)
			else:
				self.initial_visual.load()
		if self.visualize_target:
			self.target_visual = VisualObject(visual_shape=p.GEOM_CYLINDER,
				rgba_color=[0, 1, 0, 0.4],
				radius=0.3,
				length=cyl_length,
				initial_offset=[0, 0, cyl_length / 2.0])
			if self.config.get('target_visible_to_agent', False) or self.record:
			# TODO: target visual is imported for better video shooting.
			# But this would cause problem if depth camera is used .
				self.simulator.import_object(self.target_visual)
			else:
				self.target_visual.load()
		if self.visualize_waypoints and 'waypoints' in self.sensor_inputs:
			self.waypoints_visual = []
			for i in range(self.num_waypoints):
				waypoint_visual = VisualObject(visual_shape=p.GEOM_CYLINDER,
					rgba_color=[1, 0, 0, 1 - 1.0 / self.num_waypoints * i],
					radius=0.05,
					length=cyl_length,
					initial_offset=[0, 0, cyl_length / 2.0])
				if self.record:
					self.simulator.import_object(waypoint_visual)
				else:
					waypoint_visual.load()
				self.waypoints_visual.append(waypoint_visual)
		if 'subgoal' in self.sensor_inputs:
			self.subgoal_visual = VisualObject(visual_shape=p.GEOM_CYLINDER,
				rgba_color=[1, 0, 0, 1],
				radius=0.1,
				length=cyl_length,
				initial_offset=[0, 0, cyl_length / 2.0])
			self.subgoal_visual.load()

	def visualize_lidar_points(self, lidar_points=None):
		try:
			self.lidar_points
		except:
			self.lidar_points = list()
			for i in range(self.n_horizontal_rays):				
				self.lidar_points.append(VisualObject(visual_shape=p.GEOM_CYLINDER,
				rgba_color=[1, 0, 0, 1.0],
				radius=0.05,
				length=0.05,
				initial_offset=[0, 0, 0.01]))
				self.lidar_points[i].load()
		
		for i in range(self.n_horizontal_rays):				
			self.lidar_points[i].set_position(lidar_points[i])

	def initialize_sensors(self, observation_space):
		print('sensor inputs: {}'.format(self.sensor_inputs))
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
		if 'subgoal' in self.sensor_inputs:
			assert 'sensor' not in self.sensor_inputs, 'subgoal and final goal should not coexist'
			self.subgoal_space = gym.spaces.Box(low=np.inf,
				high=np.inf,
				shape=(2,),
				dtype=np.float32)
			observation_space['sensor'] = self.subgoal_space
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
			print('initialize scan')
			robot_to_scan_config = self.config.get('robot_to_scan_config', 'configs/robot_to_scan.yaml')
			with open(robot_to_scan_config, 'r') as configfile:
				robot_to_scan = yaml.load(configfile)
			print(robot_to_scan)
			self.scan_name = robot_to_scan[self.robot_name]
			self.scan_mode = self.config.get('scan_mode', 'xyz')
			scan_channels = 1 if self.scan_mode == 'dist' else 3
			# self.lidar_buffer = np.ones((self.n_horizontal_rays * self.n_vertical_beams, scan_channels)) * 30
			# self.scan_space = gym.spaces.Box(low=-np.inf, 
			# 	high=np.inf,
			# 	shape=(self.n_horizontal_rays * self.n_vertical_beams, scan_channels * 2),
			# 	dtype=np.float32)
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

	
	def get_auxiliary_sensor(self, collision_links=[]):
		return np.array([])
	

	#########################################
	# Reset Helpers.
	#########################################
	def reset_variables(self):
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
		reset_complete = False
		while not reset_complete:
			# print("Resetting robot state!!!!")
			initial = random.choice(self.agent_sources)
			initial_x = self.locations[initial]['center_x']
			initial_y = self.locations[initial]['center_y']
			x_range_radius = self.locations[initial]['x_range_radius']
			y_range_radius = self.locations[initial]['y_range_radius']
			self.initial_pos = np.array([self._get_random_point(initial_x, initial_y, 
				x_range_radius, y_range_radius), self.robot_height])[0]
			print("INITIAL POSE!!!!!!!!!!!!!", self.initial_pos)
			self.robots[0].set_position(pos=self.initial_pos)
			self.robots[0].set_orientation(
				orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz'))
			# Check for collision.
			collision_links = self.run_simulation(keep_trajectory=False)
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
			self.current_target_position = np.array([self._get_random_point(target_x, target_y,
				x_range_radius, y_range_radius), 0.0])
			dist = l2_distance(self.initial_pos[:2], self.current_target_position[:2])
			reset_complete = dist >= 1.0

		# Compute waypoints for new path.
		# print('reset! global planner: {}'.format(self.global_planner))
		if self.global_planner and self.reward_type == 'geodesic':
			self.waypoint_resolution = self.config.get('waypoint_resolution', 0.5)
			self.waypoints, self.shortest_distance = self.compute_waypoints(self.initial_pos[:2], self.current_target_position[:2])
		else:
			self.shortest_distance = l2_distance(self.current_target_position[:2], self.get_position_of_interest()[:2])
		# print('number of waypoints: {}'.format(self.waypoints))

		# Semi-hard code agent's current location.
		if initial.startswith('h'):
			self.agent_location = 'hallway'
		elif initial.startswith('c'):
			self.agent_location = 'crossing'
		elif initial.startswith('d'):
			self.agent_location = 'doorway'

			
	def reset_other_objects(self):
		pass

	def get_agent_location(self):
		return self.agent_location


	def compute_waypoints(self, source, target):
		# print('compute waypoints!')
		assert self.global_planner is not None, "Global planner is not available"
		path, geodesic_distance = self.global_planner.get_shortest_path(source, target)
		return path, geodesic_distance


	def get_closest_waypoints(self):
		assert self.global_planner is not None, "Global planner is not available"
		self.rob_pos = self.robots[0].get_position()
		closest_idx = np.argmin(np.linalg.norm(self.rob_pos[:2] - self.waypoints, axis=1))
		closest_waypoints = self.waypoints[closest_idx : closest_idx + self.num_waypoints].copy()
		num_remaining_waypoints = self.num_waypoints - closest_waypoints.shape[0]
		if num_remaining_waypoints > 0:
			remaining_waypoints = np.tile(self.current_target_position[:2], (num_remaining_waypoints, 1))
			closest_waypoints = np.concatenate((closest_waypoints, remaining_waypoints), axis=0)
		return closest_waypoints

	#########################################
	# State Getters.
	#########################################
	def get_sensor_states(self, state, collision_links=[]):
		# Omit 'normal' and 'seg' 'depth_seg', for now because we are not using them.
		if 'sensor' in self.sensor_inputs:
			state['sensor'] = self.get_additional_states()
		if 'auxiliary_sensor' in self.sensor_inputs:
			state['auxiliary_sensor'] = auxiliary_sensor
		if 'pointgoal' in self.sensor_inputs:
			state['pointgoal'] = self.get_additional_states()[:2]
		if 'rgb' in self.sensor_inputs:
			state['rgb'] = self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]
		if 'depth' in self.sensor_inputs:
			depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
			state['depth'] = depth
		if 'subgoal' in self.sensor_inputs:
			closest_waypoints = self.get_closest_waypoints()
			self.subgoal_visual.set_position([closest_waypoints[-1], 0.0])
			state['sensor'] = self._relative_to_robot(closest_waypoints)[-1]
		if 'rgb_filled' in self.sensor_inputs:
			with torch.no_grad():
				tensor = transforms.ToTensor()((state['rgb'] * 255).astype(np.uint8)).cuda()
				rgb_filled = self.comp(tensor[None, :, :, :])[0].permute(1, 2, 0).cpu().numpy()
				state['rgb_filled'] = rgb_filled
		if 'scan' in self.sensor_inputs:
			assert self.scan_name in self.robots[0].parts, "Requested scan but no scan_link"
			state['scan'] = self._get_lidar_state()
		if 'waypoints' in self.sensor_inputs :
			# print('waypoints!')
			# path = self.compute_a_star(self.config['scene']) # current dim is (107, 2), varying by scene and start/end points
			# rob_pos = self.robots[0].get_position()
			# path_robot_relative_pos = [[path[i][0] - rob_pos[0], path[i][1] - rob_pos[1]] for i in range(path.shape[0])]
			# path_robot_relative_pos = np.asarray(path_robot_relative_pos)
			# path_point_ind = np.argmin(np.linalg.norm(path_robot_relative_pos , axis=1))
			# curr_points_num = path.shape[0] - path_point_ind
			# keep the dimenstion based on the number of waypoints specified in the config file
			# if curr_points_num > self.config['waypoints']:
			# 	out = path_robot_relative_pos[path_point_ind:path_point_ind+self.config['waypoints']]
			# else:
			# 	curr_waypoints = path_robot_relative_pos[path_point_ind:]
			# 	end_point = np.repeat(path_robot_relative_pos[path.shape[0]-1].reshape(1,2), (self.config['waypoints']-curr_points_num), axis=0)
			# 	out = np.vstack((curr_waypoints, end_point)
			# state['waypoints'] = out.flatten()
			# Reset waypoint visual object.
			# self.waypoints = self.compute_waypoints(self.robots[0].get_position()[:2], self.current_target_position[:2])
			# closest_waypoints = self.get_closest_waypoints()
			# Reschedule after ~6m under max linear vel = 0.3m/s.
			if self.current_step % self.replan_interval == 0 and self.current_step > 0:
				self.waypoints, _ = self.compute_waypoints(self.robots[0].get_position()[:2], self.current_target_position[:2])

			closest_waypoints = self.get_closest_waypoints()
			for i in range(self.num_waypoints):
				self.waypoints_visual[i].set_position([closest_waypoints[i], 0.0])
			closest_waypoints = self._relative_to_robot(closest_waypoints)
			state['waypoints'] = closest_waypoints.flatten()


	def _relative_to_robot(self, pos):
		robot_pos = self.robots[0].get_position()
		if len(pos.shape) == 1:
			if pos.shape[0] == 2:
				pos = np.array([pos, robot_pos[2]])
			relative_pos = pos - robot_pos
			rotated_pos = rotate_vector_3d(relative_pos, *self.robots[0].get_rpy())
			return rotated_pos[0:2]
		elif len(pos.shape) == 2:
			pos = np.concatenate([pos, robot_pos[2] * np.ones((pos.shape[0], 1))], axis=1)
			relative_pos = pos - robot_pos
			rotated_pos = rotate_vector_3d(relative_pos.T, *self.robots[0].get_rpy()).T
			return rotated_pos[:, 0:2]




	def get_additional_states(self):
		# relative_position = self.current_target_position - self.robots[0].get_position()
		# rotate relative position back to body point of view
		# additional_states = rotate_vector_3d(relative_position, *self.robots[0].get_rpy())
		# additional_states = additional_states[0:2]
		additional_states = self._relative_to_robot(self.current_target_position)

		if self.config['task'] == 'reaching':
			end_effector_pos = self.robots[0].get_end_effector_position() - self.robots[0].get_position()
			end_effector_pos = rotate_vector_3d(end_effector_pos, *self.robots[0].get_rpy())
			additional_states = np.concatenate((additional_states, end_effector_pos))
		assert len(additional_states) == self.additional_states_dim, 'additional states dimension mismatch'

		return additional_states

	
	def _get_lidar_state(self):
		if self.config['robot'] == 'TurtlebotDifferentialDrive':
			# Hokuyo URG-04LX-UG01
			laser_linear_range = 5.6
			laser_angular_range = 360.0
			min_laser_dist = 0.05
			laser_link_name = 'scan_link'
		elif self.config['robot'] in ['JR2', 'JR2DifferentialDrive']:
			# SICK TiM571-2050101 Laser Range Finder
			laser_linear_range = 25.0
			laser_angular_range = 360.0
			#laser_angular_range = 194.806 # front lidar only
			min_laser_dist = 0.05
			laser_link_name = 'front_laser_link'
		elif self.config['robot'] == 'PatricksTurtlebot':
			# Hokuyo URG-04LX-UG01
			laser_linear_range = 5.6
			laser_angular_range = 195
			min_laser_dist = 0.02
			laser_link_name = 'scan_link'
		else:
			assert False, 'unknown robot for LiDAR observation'
			
		laser_angular_range *= np.pi / 180.0

		pose_camera = self.robots[0].parts[laser_link_name].get_pose()
		angle = np.arange(-laser_angular_range/2.0, laser_angular_range/2.0, laser_angular_range / float(self.n_horizontal_rays))
		camera_orientation = self.robots[0].parts[laser_link_name].get_orientation()

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
		results = p.rayTestBatch(pose_camera, pose_camera + offset * laser_linear_range, 6)
		# print('results: {}'.format(results[0]))
		hit = np.array([item[0] for item in results])
		dist = np.array([item[2] for item in results])
		valid_pts = (dist < 1. - 1e-5) & (dist > min_laser_dist / laser_linear_range) & (hit != self.robots[0].robot_ids[0]) & (hit != -1)
		if self.scan_mode == 'dist':
			dist[~valid_pts] = 1.0
			dist *= laser_linear_range 
			lidar_output = np.hstack([dist, self.lidar_buffer])
			self.lidar_buffer = dist
			return lidar_output

		if self.scan_mode == 'xyz':
			dist[~valid_pts] = 1.0
			dist *= laser_linear_range
			lidar_scan = np.expand_dims(dist, 1) * orig_offset
			
			if self.visualize_lidar:
				camera_rotation = Rotation.from_quat(camera_orientation)
				self.visualize_lidar_points(camera_rotation.apply(lidar_scan) + pose_camera)
				  
			return lidar_scan
# =======
# 			if self.sensor_noise:
# 				noise = np.random.normal(size=lidar_scan.shape)
# 				lidar_scan += noise
# 			# lidar_output = np.hstack([lidar_scan, self.lidar_buffer])
# 			# self.lidar_buffer = lidar_scan
# 			lidar_output = lidar_scan
# 			return lidar_output
# >>>>>>> dev
	

	#########################################
	# Reward Getters.
	#########################################
	def get_core_reward(self, collision_links=[], action=None, info={}):
		# slack reward: penalty for each timestep.
		reward = self.slack_reward
		self.potential = self.get_potential()

		if self.potential_reward_weight > 0:
			new_normalized_potential = self.potential / self.initial_potential
			potential_reward = self.normalized_potential - new_normalized_potential
			reward += potential_reward * self.potential_reward_weight
			self.normalized_potential = new_normalized_potential

		# electricity reward:
		if self.electricity_reward_weight != 0:
			#electricity_reward = np.abs(self.robots[0].joint_speeds * self.robots[0].joint_torque).mean().item()
			linear_velocity = self.robots[0].get_velocity()
			angular_velocity = self.robots[0].get_angular_velocity()[2]
			electricity_reward = np.linalg.norm(linear_velocity) #+ np.linalg.norm(angular_velocity)
			reward += electricity_reward * self.electricity_reward_weight
			#print("ELECTRICITY:", electricity_reward, electricity_reward * self.electricity_reward_weight)

		# stall torque reward:
		if self.stall_torque_reward_weight > 0:
			stall_torque_reward = np.square(self.robots[0].joint_torque).mean()
			reward += stall_torque_reward * self.stall_torque_reward_weight

		# collision reward.
		self.collision = self.detect_collision_for_reward(collision_links)
		reward += int(self.collision) * self.collision_reward_weight

		# goal reward.
		if self.potential < self.dist_tol:
			reward += self.success_reward

		return reward, info


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
		info['non_pedestrian_collision_rate'] = 0 if self.current_episode == 0 else 100 * (self.n_collisions - self.n_pedestrian_collisions) / (self.current_episode)
		info['pedestrian_collision_rate'] = 0 if self.current_episode == 0 else 100 * self.n_pedestrian_collisions / (self.current_episode)
		info['robot_hit_pedestrian_rate'] = 0 if self.current_episode == 0 else 100 * self.n_robot_hit_pedestrian / (self.current_episode)
		info['pedestrian_hit_robot_rate'] = 0 if self.current_episode == 0 else 100 * self.n_pedestrian_hit_robot / (self.current_episode)
		info['timeout_rate'] = 0 if self.current_episode == 0 else 100 * self.n_timeouts / (self.current_episode)
		info['mean_distance_to_nearest_pedestrian'] = 0 if self.current_episode == 0 else mean(self.log_distance_to_nearest_pedestrian)
		#info['variance_distance_to_nearest_pedestrian'] = 0 if self.current_episode == 0 else variance(self.log_distance_to_nearest_pedestrian)
		info['path_length'] = 0
		info['return'] = self.total_reward
		info['shortest_path_length'] = 0 if self.current_episode == 0 else [self.spl_mean]
		info['time_elapsed'] = 0 if (self.current_episode == 0 or self.n_successes == 0) else [self.time_elapsed / self.n_successes]
		info['ave_jerk'] = 0 if self.current_episode == 0 else [self.total_jerk / self.current_episode]


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
		if self.reward_type == 'l2':
			dist = l2_distance(self.current_target_position[:2], self.get_position_of_interest()[:2])
		elif self.reward_type == 'geodesic':
			print("HELLO!!!!")
			self.waypoints, dist = self.compute_waypoints(self.robots[0].get_position()[:2], \
				self.current_target_position[:2]) 
		return dist


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

	def render(self, mode='human'):
		if mode == 'human':
			pass # not sure what to use here.
		if mode == 'rgb_array':
			# robot focus
			robot_pos = self.robots[0].get_position()
			"""
			robot_XYZW = self.robots[0].get_orientation()
			quat = quatFromXYZW(robot_XYZW, 'xyzw')
			yaw = np.arctan2(2.0 * (quat[1] * quat[2] + quat[3] * quat[0]), \
				quat[3] ** 2 - quat[0] ** 2 - quat[1] ** 2 + quat[2] ** 2) + 45.0
			distance = 6.0
			pitch = self.camera_pitch
			roll = 0.0
			view_matrix = p.computeViewMatrixFromYawPitchRoll(robot_pos[:3], distance, \
				yaw, pitch, roll, upAxisIndex=2)
			print('numpy enabled: {}'.format(p.isNumpyEnabled()))
			w, h, img, depth, seg = p.getCameraImage(width=512, height=512, viewMatrix=view_matrix)
			print('shape of img: {} - {}'.format(len(img), type(img)))
			"""
			# camera_pos = robot_pos[:3] + np.array([0, 0, 6])
			# up = [0, 0, 0.5]
			# TODO: Hard coded eye position -> pitch / yaw / roll conversion.
			self.simulator.renderer.set_camera(camera=robot_pos[:3]+np.array([4.0, 0.1, 8.0]), \
				target=robot_pos[:3], up=[0, 0, 1])
			# elf.simulator.renderer.set_camera(camera=[0.0, 0.0, 30.0], \
			# 	target=robot_pos[:3], up=[0, 0, 1])
			img = self.simulator.renderer.render(modes=('rgb'))
			# print('image: {}'.format(img[0].shape))
			img = (img[0][:, :, :3] * 255).astype(np.uint8)
			# print(img)
			# print('shape afterwards: {}'.format(img.shape))
			return img
			# return np.zeros((512, 512, 3), dtype=np.uint8)



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
		deviation = np.random.random_sample(3) * 2 - 1
		center = np.array([x, y, 0])
		deviation = np.array([range_x, range_y, 0]) * deviation
		
		return center + deviation


	# ==================== Code for video shooting ============== #
	def capture_frame(self):
		frame = self.render(mode='rgb_array')
		# assert frame is None, "frame is None"
		self.last_frame = frame
		self._encode_frame(frame)


	def _encode_frame(self, frame):
		if not self.encoder:
			self.encoder = ImageEncoder(self.record_path, frame.shape, self.output_frames_per_sec)
			self.record_metadata['encoder_version'] = self.encoder.version_info
		try:
			self.encoder.capture_frame(frame)
		except error.InvalidFrame as e:
			print('Tried to pass invalid frame, marking ad broken %s', e)


	def start_record(self):
		self.video_folder = os.path.join(self.root_dir, self.video_folder)
		print('video folder: {}'.format(self.video_folder))
		if not os.path.exists(self.video_folder):
			os.makedirs(self.video_folder)
		self.video_name = str(datetime.now())
		self.record_metadata = dict()
		self.encoder = None
		self.step_id = 0
		


	def start_video_recorder(self):
		self.close_video_recorder()
		video_name = '{}-episode-{}-step-{}-to-step-{}'.format(self.video_name, self.current_episode, self.step_id, \
            self.step_id + self.video_episode_length)
		base_path = os.path.join(self.video_folder, video_name)
		self.record_path = base_path + '.mp4'
		self.recorded_frames = 1
		# self.recording = True


	def close_video_recorder(self):
		# self.recording = False
		if self.encoder:
			self.encoder.close()
			self.encoder = None


	def _video_start(self):
		# print('current episode: {} step {}'.format(self.current_episode, self.current_step))
		return self.current_episode % 100 == 1 and self.current_step == 1

	def _video_end(self):
		return self.step_id > self.video_length or self.current_episode %  100 == 0


	def _new_video_episode(self):
		return self.step_id % self.video_episode_length == 0