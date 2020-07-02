from core.pedestrians.state import ObservableState
from core.pedestrians.action import ActionXY

import random
import numpy as np
import pybullet as p
import gym

from gibson2.core.physics.scene import BuildingScene, StadiumScene
from core.objects import BoxShape
from gibson2.utils.utils import rotate_vector_3d, l2_distance, quatToXYZW, parse_config
from collections import OrderedDict, namedtuple
from core.nav_env import NavigateEnv
from core.pedestrian_model import Pedestrian
from transforms3d.euler import euler2quat

from core.objects import VisualObject
from utils.global_planner import GlobalPlanner

import itertools
import os

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
	'timeout',
	'personal_space_violations',
#	'average_personal_space',
#	'min_personal_space',#
	'robot_trajectories',
	'pedestrian_trajectories',
	])

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
		verbose=False,
		global_planner=None,
		record=False,
		**kwargs
	):
		
		super().__init__(config_file,
			mode=mode,
			action_timestep=action_timestep,
			physics_timestep=physics_timestep,
			automatic_reset=automatic_reset,
			device_idx=device_idx,
			verbose=verbose,
			global_planner=global_planner,
			record=record,
			**kwargs)

		# Pedestrian-related metrics.
		self.n_personal_space_violations = 0
		self.n_robot_hit_pedestrian = 0
		self.n_pedestrian_hit_robot = 0
		self.n_pedestrian_collisions = 0
		self.n_cutting_off = 0
		self.total_personal_space = 0
		self.min_personal_space = 10 ** 5
		self.pedestrian_trajectories = {}

		self.pedestrian_chats = {}
		self.pedestrian_last_chat = {}
		self.distance_to_nearest_pedestrian = []

		self.pedestrians_actions = dict()

	def load_parameters(self):
		super().load_parameters()
		self.num_obstacles = self.config.get('num_obstacles', 0)
		self.num_pedestrians = self.config.get('num_pedestrians', {})
		self.randomize_pedestrian_attributes = self.config.get('randomize_pedestrian_attributes', False)
		self.pedestrian_height = 0.03
		pedestrians_dynamics = self.config.get('pedestrians_dynamics')
		self.total_num_pedestrians = sum([int(num_ped) for _, num_ped in self.num_pedestrians.items()])
		self.pedestrians_radius = pedestrians_dynamics['radius']
		self.robot_radius = self.config.get('robot_radius', 0.3)
		
		self.use_pedestrian_behaviors = self.config['pedestrians_dynamics'].get('use_pedestrian_behaviors', False)

		self.chat_probability = self.config['pedestrians_dynamics'].get('chat_probability', 0.0)
		self.chat_min_delay = self.config['pedestrians_dynamics'].get('chat_min_delay', 0.0)
		self.ped_waypoints = self.room_connectivity['ped-waypoints']

		# self.pedestrians_x_range_radius = self.movements['pedestrians']['x_range_radius']
		# self.pedestrians_y_range_radius = self.movements['pedestrians']['y_range_radius']
		self.personal_space = pedestrians_dynamics['personal_space']
		self.pedestrians_min_separation = (pedestrians_dynamics['radius'] + self.personal_space) * 2
		self.pedestrians_dist_tol = pedestrians_dynamics['dist_tol']
		self.pedestrians_can_see_robot = self.config.get('pedestrians_can_see_robot', False)
		self.psv_reward_weight = self.config.get('psv_reward_weight', 0)
		self.robot_pedestrians_min_separation = self.config.get('robot_pedestrians_min_separation', 2.0)
		print('PEDESTRIAN CAN SEE ROBOT: {}'.format(self.pedestrians_can_see_robot))
		print('\n' * 5)
		self.walls = self.config.get('walls', None)
		self.save_trajectories = self.config.get('save_trajectories', False)

		self.use_crowdnav_scenario = self.config.get('use_crowdnav_scenario', False)
		self.crowdnav_circle_radius = self.config.get('crowdnav_circle_radius', 4.0)
		self.crowdnav_square_width = self.config.get('crowdnav_square_width', 10.0)

		self.global_time = 0
		
		self.current_target_position = np.array([-100, -100, 0])

	def build_scene(self):
		# Build wall.
		if self.walls:
			for wall_pos, wall_dim in zip(self.walls['walls_pos'], self.walls['walls_dim']):
				wall = BoxShape(pos=wall_pos, dim=wall_dim, rgba_color=[1.0, 1.0, 1.0, 0.6], mass=0)
				self.obstacle_ids.append(self.simulator.import_object(wall))
				
		# List of Gibson pedestrian IDs
		self.pedestrian_gibson_ids = []
		
		# Load pedestrians
		self.load_pedestrians()

	
	def initialize_sensors(self, observation_space):
		super().initialize_sensors(observation_space)
		# TODO: Check if pedestrian position is given relative to the robot.
		if 'pedestrian_position' in self.sensor_inputs:
			self.pedestrian_position_space = gym.spaces.Box(low=-np.inf, high=np.inf,
				shape=(self.total_num_pedestrians * 2,),
				dtype=np.float32)
			observation_space['pedestrian_position'] = self.pedestrian_position_space
		if 'pedestrian_velocity' in self.sensor_inputs:
			self.pedestrian_velocity_space = gym.spaces.Box(low=-np.inf, high=np.inf,
				shape=(self.total_num_pedestrians * 2,),
				dtype=np.float32)
			observation_space['pedestrian_velocity'] = self.pedestrian_velocity_space
		if 'pedestrian_ttc' in self.sensor_inputs:
			self.pedestrian_position_space = gym.spaces.Box(low=-np.inf, high=np.inf,
				shape=(self.total_num_pedestrians,), 
				dtype=np.float32)
			observation_space['pedestrian_ttc'] = self.pedestrian_position_space


	def reset_variables(self):
		super().reset_variables()
		self.total_personal_space = 0
		self.min_personal_space = 10 ** 5
		#self.n_personal_space_violations = 0
		self.robot_trajectories = []
		self.pedestrian_trajectories = {}
		self.distance_to_nearest_pedestrian = np.inf
		self.episode_distance = 0.0
		
	def reset_other_objects(self):
		if self.use_crowdnav_scenario:
			if self.num_pedestrians['crowdnav_circle_crossing'] != 0:
				self.agent_location = 'crowdnav_circle_crossing'
				self.generate_circle_crossing_humans()
			else:
				self.agent_location = 'crowdnav_square_crossing'
				self.generate_square_crossing_humans()

	def reset_agent(self):
		if self.use_crowdnav_scenario:
			self.reset_agent_crowdnav()
		else:
			self.reset_agent_svl()

		self.pedestrians_actions = dict()


	# Overwrite reset_agent to check distance between robot & pedestrians.
	def reset_agent_svl(self):
		self.robots[0].robot_specific_reset()

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
			reset_complete = self._no_collision(collision_links) and self._distance_to_closest_pedestrian() > self.robot_pedestrians_min_separation

		# Reset robot's target position.
		#TODO: check if target collides with a static obstacle?
		target = random.choice(self.room_connectivity[initial])
		target_x = self.locations[target]['center_x']
		target_y = self.locations[target]['center_y']
		x_range_radius = self.locations[target]['x_range_radius']
		y_range_radius = self.locations[target]['y_range_radius']
		reset_complete = False
		while not reset_complete:
			self.current_target_position = np.array([*self._get_random_point(target_x, target_y,
				x_range_radius, y_range_radius), 0.0])
			dist = l2_distance(self.initial_pos[:2], self.current_target_position[:2])
			reset_complete = dist >= 1.0
			
		if self.global_planner:
			self.waypoint_resolution = self.config.get('waypoint_resolution', 0.5)
			self.waypoints, self.shortest_distance = self.compute_waypoints(self.initial_pos[:2], self.current_target_position[:2])
		else:
			self.shortest_distance = l2_distance(self.initial_pos[:2], self.current_target_position[:2])
		
		# Semi-hard code agent's current location.
		if initial.startswith('h'):
			self.agent_location = 'hallway'
		elif initial.startswith('c'):
			self.agent_location = 'crossing'
		elif initial.startswith('d'):
			self.agent_location = 'doorway'
			
	def reset_agent_crowdnav(self):
		self.robots[0].robot_specific_reset()
		
		# CrowdNav always uses the same intial position and goal for the robot
		px = 0
		
		if self.agent_location == 'crowdnav_circle_crossing':
			py = -self.crowdnav_circle_radius
		else:
			py = -self.crowdnav_square_width / 2.0
		gx = -px
		gy = -py

		self.initial_pos = np.array([px, py, 0.0])
		self.robots[0].set_position(pos=self.initial_pos)
		orientation = np.arctan2(gy - py, gx - px)
		self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(0, 0, orientation), 'wxyz'))
		self.current_target_position = np.array([gx, gy, 0])
		
		if self.global_planner:
			self.waypoint_resolution = self.config.get('waypoint_resolution', 0.5)
			self.waypoints, self.shortest_distance = self.compute_waypoints(self.initial_pos[:2], self.current_target_position[:2])
		else:
			self.shortest_distance = l2_distance(self.initial_pos[:2], self.current_target_position[:2])

		if self.keep_trajectory:
			trajectory_keys = list(self.trajectory.keys())
			for key in trajectory_keys:
				ped_key = 'ped_{}'.format(key)
				self.trajectory[ped_key] = dict()
				for location in self.pedestrians:
					local_pedestrians = self.pedestrians[location]
					for pid in local_pedestrians:
						self.trajectory[ped_key][pid] = []


	##################################################
	# Pedestrian loading / resetting # 
	##################################################
	

	def create_pedestrian(self, pos, orn, target, location, target_location):
		pedestrian = Pedestrian(config=self.config, simulator=self.simulator, location=location, target_location=target_location,
			pos=pos, orn=orn, target=target, visual_meshes=True, time_step=self.physics_timestep)
				
		if self.randomize_pedestrian_attributes:
			pedestrian.sample_random_attributes()
			
		pedestrian.create(pos)
		self.pedestrian_gibson_ids.append(pedestrian.gibson_id)
		return pedestrian
	
	def load_pedestrians(self):
		if self.use_crowdnav_scenario:
			self.load_pedestrians_crowdnav()
		else:
			self.load_pedestrians_svl()

	def load_pedestrians_crowdnav(self):
		self.pedestrians = {}
		
		if self.num_pedestrians['crowdnav_circle_crossing'] != 0:
			self.agent_location = 'crowdnav_circle_crossing'
			self.generate_circle_crossing_humans(create_pedestrians=True)
		else:
			print("LOADING PEDS!!!!!!!!")
			self.agent_location = 'crowdnav_square_crossing'
			self.generate_square_crossing_humans(create_pedestrians=True)
				
		return
				
		location_keys = self.num_pedestrians.keys()

		for location in location_keys:
			num_pedestrian = self.num_pedestrians[location]
			for i in range(num_pedestrian):
				reset_complete = False
				# Find initial position.
				while not reset_complete:
						
					#goal_marker = self.pedestrians[location][pid]['goal_marker']
					#new_initial = self.pedestrians[location][pid]['target'] 
					
					angle = np.random.random() * np.pi * 2
					# add some noise to simulate all the possible cases robot could meet with human
					px_noise = (np.random.random() - 0.5) #* human.v_pref
					py_noise = (np.random.random() - 0.5) #* human.v_pref
					px = self.crowdnav_circle_radius * np.cos(angle) + px_noise
					py = self.crowdnav_circle_radius * np.sin(angle) + py_noise
					initial_pos = np.array([px, py, 0.0])
					
					collide = False
					for used_pos in used_sources_pos:
						if l2_distance(used_pos[:2], initial_pos[:2]) < (self.pedestrians_radius + self.robot_radius + self.personal_space):
							collide = True
							break
					
					# Find target position.
					if not collide:
#						 angle = np.random.random() * np.pi * 2
#						 # add some noise to simulate all the possible cases robot could meet with human
#						 gx_noise = (np.random.random() - 0.5) #* human.v_pref
#						 gy_noise = (np.random.random() - 0.5) #* human.v_pref
#						 gx = self.crowdnav_circle_radius * np.cos(angle) + gx_noise
#						 gy = self.crowdnav_circle_radius * np.sin(angle) + gy_noise
						
						gx = -px
						gy = -py

						target_pos = np.array([gx, gy, 0.0])
						
						theta = np.arctan2(target_pos[1] - initial_pos[1], target_pos[0] - initial_pos[0])
						orn = p.getQuaternionFromEuler([0, 0, theta])
						
						pedestrian = self.create_pedestrian(initial_pos, orn, target_pos, location=location, target_location=location)

						goal_marker = VisualObject(visual_shape=p.GEOM_CYLINDER,
							rgba_color=[0.0, 1.0, 1.0, 0.6],
							radius=0.05,
							length=1.0,
							initial_offset=[0.0, 0.0, 0.5])
						goal_marker.load()
						goal_marker.set_position(target_pos)
						
						if self.randomize_pedestrian_attributes:
							pedestrian.sample_random_attributes()
						
						if location not in self.pedestrians:
							self.pedestrians[location] = dict()
						self.pedestrians[location][pedestrian.gibson_id] = {'object': pedestrian, \
						'initial': initial_pos, 'target': target_pos, 'goal_marker': goal_marker, 'stop_timestamp': 0}
						
						used_sources_pos.append(initial_pos)
						reset_complete = True
				
				if self.use_pedestrian_behaviors:
					pedestrian.pause_start_time = self.global_time
					pedestrian.pause_interval = np.random.uniform(pedestrian.min_pause_interval, pedestrian.max_pause_interval)

					pedestrian.chat_interval = np.random.uniform(pedestrian.chat_interval_min, pedestrian.chat_interval_max)
					pedestrian.chat_distance = np.random.uniform(pedestrian.chat_distance_min, pedestrian.chat_distance_max)
	
	def load_pedestrians_svl(self):
		self.pedestrians = {}
		used_sources_pos = []

		# Skip CrowdNav layouts if using our own
		try:
			del self.num_pedestrians['crowdnav_circle_crossing']
			del self.num_pedestrians['crowdnav_square_crossing']
		except:
			pass
		
		location_keys = self.num_pedestrians.keys()
		for location in location_keys:
			num_pedestrian = self.num_pedestrians[location]
			for i in range(num_pedestrian):
				reset_complete = False
				# Find initial position.
				while not reset_complete:
					initial = random.choice(self.movements['pedestrians'][location]['sources'])
					target = random.choice(self.room_connectivity[initial])
					
					initial_x = self.locations[initial]['center_x']
					initial_y = self.locations[initial]['center_y']
					target_x = self.locations[target]['center_x']
					target_y = self.locations[target]['center_y']
					x_range_radius = self.locations[initial]['x_range_radius']
					y_range_radius = self.locations[initial]['y_range_radius']
					
					initial_pos = np.array([*self._get_random_point(initial_x, initial_y,\
					x_range_radius, y_range_radius), self.pedestrian_height])
					
					collide = False
					for used_pos in used_sources_pos:
						if l2_distance(used_pos[:2], initial_pos[:2]) < self.pedestrians_min_separation:
							collide = True
							break
					
					# Find target position.
					if not collide:
						x_range_radius = self.locations[target]['x_range_radius']
						y_range_radius = self.locations[target]['y_range_radius']
						target_pos = np.array([*self._get_random_point(target_x, target_y, 
							x_range_radius, y_range_radius), 0.0])
						theta = np.arctan2(target_pos[1] - initial_pos[1], target_pos[0] - initial_pos[0])
						orn = p.getQuaternionFromEuler([0, 0, theta])
						pedestrian = self.create_pedestrian(initial_pos, orn, target_pos, location=location, target_location=target)
						goal_marker = VisualObject(visual_shape=p.GEOM_CYLINDER,
							rgba_color=[0.0, 1.0, 1.0, 0.6],
							radius=0.05,
							length=1.0,
							initial_offset=[0.0, 0.0, 0.5])
						goal_marker.load()
						goal_marker.set_position(target_pos)
						
						if self.randomize_pedestrian_attributes:
							pedestrian.sample_random_attributes()
						
						if location not in self.pedestrians:
							self.pedestrians[location] = dict()
						self.pedestrians[location][pedestrian.gibson_id] = {'object': pedestrian, \
						'initial': initial, 'target': target, 'goal_marker': goal_marker, 'stop_timestamp': 0}
						
						used_sources_pos.append(initial_pos)
						reset_complete = True
						
				if self.use_pedestrian_behaviors:
					pedestrian.pause_start_time = self.global_time
					pedestrian.pause_interval = np.random.uniform(pedestrian.min_pause_interval, pedestrian.max_pause_interval)
	
					pedestrian.chat_interval = np.random.uniform(pedestrian.chat_interval_min, pedestrian.chat_interval_max)
					pedestrian.chat_distance = np.random.uniform(pedestrian.chat_distance_min, pedestrian.chat_distance_max)


	def reset_single_pedestrian(self, location, pid):
		pedestrian = self.pedestrians[location][pid]['object']
		
		if self.randomize_pedestrian_attributes:
			pedestrian.sample_random_attributes()
			
		goal_marker = self.pedestrians[location][pid]['goal_marker']
		new_initial = self.pedestrians[location][pid]['target'] 
		new_target = random.choice(self.room_connectivity[new_initial])
		pedestrian.target_location = new_target
		self.pedestrians[location][pid]['initial'] = new_initial
		self.pedestrians[location][pid]['target'] = new_target
		target_x = self.locations[new_target]['center_x']
		target_y = self.locations[new_target]['center_y']
		x_range_radius = self.locations[new_target]['x_range_radius']
		y_range_radius = self.locations[new_target]['y_range_radius']
		target_pos = np.array([*self._get_random_point(target_x, target_y, x_range_radius, y_range_radius), 0.0])
		pedestrian.reset_target(target_pos[0], target_pos[1])
		goal_marker.set_position(target_pos)


	def generate_circle_crossing_humans(self, location='crowdnav_circle_crossing', create_pedestrians=False):
		if self.randomize_pedestrian_attributes:
			pedestrian.sample_random_attributes()
			
		robot_pos = self.robots[0].get_position()
		
		min_separation = self.pedestrians_radius + self.robot_radius + self.personal_space
		
		initial_positions = list()
		goal_positions = list()

		for i in range(self.num_pedestrians[self.agent_location]):
			if not create_pedestrians:
				pid = list(self.pedestrians[location].keys())[i]
				pedestrian = self.pedestrians[location][pid]['object']

			
			while True:
				angle = np.random.random() * np.pi * 2
				
				# add some noise to simulate all the possible cases robot could meet with human
				px_noise = (np.random.random() - 0.5) #* human.v_pref
				py_noise = (np.random.random() - 0.5) #* human.v_pref
				px = self.crowdnav_circle_radius * np.cos(angle) + px_noise
				py = self.crowdnav_circle_radius * np.sin(angle) + py_noise
				
				gx = -px
				gy = -py
				
				test_pos = np.array([px, py, 0])
				goal_pos = np.array([gx, gy, 0])
				
				collide = False
				
				if len(initial_positions) == 0:
					if l2_distance(test_pos, robot_pos) < self.robot_pedestrians_min_separation:
						collide = True
				else:
					for initial_position in initial_positions:
						if l2_distance(test_pos, initial_position) < min_separation:
							collide = True
							break
						if l2_distance(test_pos, robot_pos) < self.robot_pedestrians_min_separation:
							collide = True
							break

				if not collide:
					initial_positions.append(test_pos)
					goal_positions.append(goal_pos)
					break
				
		if create_pedestrians:
			for i in range(len(initial_positions)):
				theta = np.arctan2(goal_positions[i][1] - initial_positions[i][1], goal_positions[i][0] - initial_positions[i][0])
				orn = p.getQuaternionFromEuler([0, 0, theta])
			
				pedestrian = self.create_pedestrian(initial_positions[i], orn, goal_positions[i], location=self.agent_location, target_location=self.agent_location)
			
				if self.randomize_pedestrian_attributes:
					pedestrian.sample_random_attributes()
			
				if self.agent_location not in self.pedestrians:
					self.pedestrians[location] = dict()
					
				self.pedestrians[location][pedestrian.gibson_id] = {'object': pedestrian, \
				'initial': initial_positions[i], 'target': goal_positions[i], 'goal_marker': None, 'stop_timestamp': 0}
		
		i = 0
		for pid in self.pedestrians[location]:
			pedestrian = self.pedestrians[location][pid]['object']
			pedestrian.set_position(initial_positions[i][0], initial_positions[i][1])
			pedestrian.gibson_object.set_position(initial_positions[i])
			pedestrian.reset_target(goal_positions[i][0], goal_positions[i][1])
			i += 1
			
	def generate_square_crossing_humans(self, location='crowdnav_square_crossing', create_pedestrians=False):
		if self.randomize_pedestrian_attributes:
			pedestrian.sample_random_attributes()
			
		robot_pos = self.robots[0].get_position()
		
		min_separation = self.pedestrians_radius + self.robot_radius + self.personal_space
		
		initial_positions = list()
		goal_positions = list()

		for i in range(self.num_pedestrians[self.agent_location]):
			if not create_pedestrians:
				pid = list(self.pedestrians[location].keys())[i]
				pedestrian = self.pedestrians[location][pid]['object']

			while True:
				if np.random.random() > 0.5:
					sign = -1
				else:
					sign = 1
					
				px = np.random.random() * self.crowdnav_square_width * 0.5 * sign
				py = (np.random.random() - 0.5) * self.crowdnav_square_width
				
				test_pos = np.array([px, py, 0])
				
				collide = False
				
				if len(initial_positions) == 0:
					if l2_distance(test_pos, robot_pos) < self.robot_pedestrians_min_separation:
						collide = True
				else:
					for initial_position in initial_positions:
						if l2_distance(test_pos, initial_position) < min_separation:
							collide = True
							break
						if l2_distance(test_pos, robot_pos) < self.robot_pedestrians_min_separation:
							collide = True
							break

				if not collide:
					initial_positions.append(test_pos)
					break
								
			while True:
				if np.random.random() > 0.5:
					sign = -1
				else:
					sign = 1
					
				gx = np.random.random() * self.crowdnav_square_width * 0.5 * -sign
				gy = (np.random.random() - 0.5) * self.crowdnav_square_width
				
				goal_pos = np.array([gx, gy, 0])
				
				collide = False
				
				if len(goal_positions) == 0:
					if l2_distance(goal_pos, self.current_target_position) < self.robot_pedestrians_min_separation:
						collide = True
				else:
					for goal_position in goal_positions:
						if l2_distance(goal_pos, goal_position) < min_separation:
							collide = True
							break
						if l2_distance(goal_pos, self.current_target_position) < self.robot_pedestrians_min_separation:
							collide = True
							break

				if not collide:
					goal_positions.append(goal_pos)
					break
								
		if create_pedestrians:
			for i in range(len(initial_positions)):
				theta = np.arctan2(goal_positions[i][1] - initial_positions[i][1], goal_positions[i][0] - initial_positions[i][0])
				orn = p.getQuaternionFromEuler([0, 0, theta])
			
				pedestrian = self.create_pedestrian(initial_positions[i], orn, goal_positions[i], location=self.agent_location, target_location=self.agent_location)
			
				if self.randomize_pedestrian_attributes:
					pedestrian.sample_random_attributes()
			
				if self.agent_location not in self.pedestrians:
					self.pedestrians[location] = dict()
					
				self.pedestrians[location][pedestrian.gibson_id] = {'object': pedestrian, \
				'initial': initial_positions[i], 'target': goal_positions[i], 'goal_marker': None, 'stop_timestamp': 0}
				
		i = 0
		for pid in self.pedestrians[location]:
			pedestrian = self.pedestrians[location][pid]['object']
			pedestrian.set_position(initial_positions[i][0], initial_positions[i][1])
			pedestrian.gibson_object.set_position(initial_positions[i])
			pedestrian.reset_target(goal_positions[i][0], goal_positions[i][1])
			i += 1
			
	def asymetric_personal_space_violation(self):
		rob_pos = np.array(self.robots[0].get_position())[:2]
		rob_vel = np.linalg.norm(self.robots[0].get_velocity()[:2])
		
		current_psv = 0

		for location in self.pedestrians:
			for pid in self.pedestrians[location]:
				curr_ped = self.pedestrians[location][pid]['object']
				ped_vel = curr_ped.get_velocity()[:2]
				rob_ped_rel_pos = rob_pos - curr_ped.get_position()[:2]
				
				psv = np.dot(ped_vel, rob_ped_rel_pos) / np.linalg.norm(rob_ped_rel_pos)
								
				#psv = max(0.0, (self.personal_space - dist) / self.personal_space)
												
				if psv > 0:
					# weight by robot velocity
					psv *= rob_vel

					# weight by time over which action takes place
					psv *= self.action_timestep
					
					current_psv += psv

		return current_psv

	def _get_robot_observable_state(self):
		px, py, pz = self.robots[0].get_position()
		theta = self.robots[0].get_rpy()[2]
		vx, vy, vz = self.robots[0].get_velocity()
		vr = self.robots[0].get_angular_velocity()[2]
		return ObservableState(px, py, theta, vx, vy, vr, self.robot_radius, self.personal_space)

	def _distance_to_closest_pedestrian(self):
		# agent_location = self.get_agent_location()
		rob_pos = np.array(self.robots[0].get_position())[:2]
		min_dist = 10 ** 5
		for location in self.pedestrians:
			for pid in self.pedestrians[location]:
				curr_ped = self.pedestrians[location][pid]['object']
				dist = l2_distance(rob_pos, curr_ped.get_position()) - (self.pedestrians_radius + self.robot_radius)
				if dist < min_dist:
					min_dist = dist
		return min_dist
	
	def _distance_to_closest_agent(self):
		# agent_location = self.get_agent_location()
		rob_pos = np.array(self.robots[0].get_position())[:2]
		min_dist = 10 ** 5
		for location in self.pedestrians:
			for pid in self.pedestrians[location]:
				curr_ped = self.pedestrians[location][pid]['object']
				dist = l2_distance(rob_pos, curr_ped.get_position()) - (self.pedestrians_radius + self.robot_radius)
				if dist < min_dist:
					min_dist = dist
		return min_dist
		
	def get_pedestrian_chats(self):
		for location in self.pedestrians:
			for ped_pair in itertools.combinations(self.pedestrians[location], 2):
				ped_1 = self.pedestrians[location][ped_pair[0]]['object']
				ped_2 = self.pedestrians[location][ped_pair[1]]['object']

				if ped_pair in self.pedestrian_chats.keys():
					if (self.global_time - self.pedestrian_chats[ped_pair]) > max(ped_1.chat_interval, ped_2.chat_interval):
						del self.pedestrian_chats[ped_pair]
						self.pedestrian_last_chat[ped_pair] = self.global_time				
				else:
					chat_distance = random.uniform(min(ped_1.chat_distance, ped_2.chat_distance), max(ped_1.chat_distance, ped_2.chat_distance))
					dist = l2_distance(ped_1.get_position(), ped_2.get_position()) - 2 * self.pedestrians_radius
	
					if dist < chat_distance:
						chat_probability = np.random.uniform(0, 1)
						if chat_probability < self.chat_probability:
							if ped_pair not in self.pedestrian_last_chat.keys() or self.global_time - self.pedestrian_last_chat[ped_pair] > self.chat_min_delay:
								self.pedestrian_chats[ped_pair] = self.global_time
	
	def get_ttc_ped(self, ped):
		ped_pos = ped.get_position()[:2]
		ped_vel = ped.get_velocity()[:2]
		
		rob_pos = self.robots[0].get_position()[:2]
		rob_vel = self.robots[0].get_velocity()[:2]
		
		rel_pos = rob_pos - ped_pos
		rel_vel = rob_vel - ped_vel
		
		if (rel_vel[0] * rel_pos[0] + rel_vel[1] * rel_pos[1]) == 0:
			time_to_collision = -1.0
		else:
			time_to_collision = -1.0 * (rel_pos[0]**2 + rel_pos[1]**2) / (rel_vel[0] * rel_pos[0] + rel_vel[1] * rel_pos[1])
			
		if time_to_collision < 0:
			time_to_collision = -1.0
			
		return time_to_collision
	
	def get_ttc(self):
		pedestrian_positions = []
		pedestrian_velocities = []
		
		for location in self.pedestrians:
			for pid in self.pedestrians[location]:
				pedestrian_positions.append(self.pedestrians[location][pid]['object'].get_position())
				pedestrian_velocities.append(self.pedestrians[location][pid]['object'].get_velocity())
				
		pedestrian_positions = np.array(pedestrian_positions)
		pedestrian_velocities = np.array(pedestrian_velocities)
		
		rob_pos = np.array(self.robots[0].get_position())[:2]
		rob_vel = np.array(self.robots[0].get_velocity())[:2]
		
		ped_rob_relative_pos = np.append((pedestrian_positions - rob_pos), \
			np.zeros((pedestrian_velocities.shape[0], 1)), 1)
		
		ped_rob_relative_vel = np.append((pedestrian_velocities - rob_vel), \
			np.zeros((pedestrian_velocities.shape[0], 1)), 1)
			
		ttc = list()
		
		for pos, vel in zip(ped_rob_relative_pos, ped_rob_relative_vel):
			if (vel[0] * pos[0] + vel[1] * pos[1]) == 0:
				time_to_collision = -1.0
			else:
				time_to_collision = -1.0 * (pos[0]**2 + pos[1]**2) / (vel[0] * pos[0] + vel[1] * pos[1])
			if time_to_collision < 0:
				time_to_collision = -1.0
			#else:
			#	time_to_collision = 1.0 - np.tanh(time_to_collision / 10.0)
			ttc.append(time_to_collision)
			
		return ttc
	
	def get_ttc_canliu(self):
		pedestrian_positions = []
		pedestrian_velocities = []
		for location in self.pedestrians:
			for pid in self.pedestrians[location]:
				pedestrian_positions.append(self.pedestrians[location][pid]['object'].get_position())
				pedestrian_velocities.append(self.pedestrians[location][pid]['object'].get_velocity())
		pedestrian_positions = np.array(pedestrian_positions)
		pedestrian_velocities = np.array(pedestrian_velocities)
		
		rob_pos = np.array(self.robots[0].get_position())[:2]
		rob_vel = np.array(self.robots[0].get_velocity())[:2]
		ped_rob_diff_buffer = np.append((pedestrian_positions - rob_pos), \
			np.zeros((pedestrian_velocities.shape[0], 1)), 1)
		ped_rob_relative_pos = np.array([rotate_vector_3d(ped_rob_diff, *self.robots[0].get_rpy())\
			for ped_rob_diff in ped_rob_diff_buffer])[:, :2]
		
		ped_rob_diff_buffer = np.append((pedestrian_velocities - rob_vel), \
			np.zeros((pedestrian_velocities.shape[0], 1)), 1)
		ped_rob_relative_vel = np.array([rotate_vector_3d(ped_rob_diff, *self.robots[0].get_rpy())\
			for ped_rob_diff in ped_rob_diff_buffer])[:, :2]
			
		ttc = list()
		
		for pos, vel in zip(ped_rob_relative_pos, ped_rob_relative_vel):
			if (vel[0] * pos[0] + vel[1] * pos[1]) == 0:
				time_to_collision = -1.0
			else:
				time_to_collision = -1.0 * (pos[0]**2 + pos[1]**2) / (vel[0] * pos[0] + vel[1] * pos[1])
			if time_to_collision < 0:
				time_to_collision = -1.0
			#else:
			#	time_to_collision = 1.0 - np.tanh(time_to_collision / 10.0)
			ttc.append(time_to_collision)
			
		return ttc
	
	def asymetric_personal_space_violation(self):
		# NOTE: still needs work...
		rob_pos = np.array(self.robots[0].get_position())[:2]
		rob_vel = np.linalg.norm(self.robots[0].get_velocity()[:2])
		
		current_psv = 0

		for location in self.pedestrians:
			for pid in self.pedestrians[location]:
				curr_ped = self.pedestrians[location][pid]['object']
				ped_vel = curr_ped.get_velocity()[:2]
				rob_ped_rel_pos = rob_pos - curr_ped.get_position()[:2]
				
				psv = np.dot(ped_vel, rob_ped_rel_pos) / np.linalg.norm(rob_ped_rel_pos)
								
				#psv = max(0.0, (self.personal_space - dist) / self.personal_space)
												
				if psv > 0:
					# weight by robot velocity
					psv *= rob_vel

					# weight by time over which action takes place
					psv *= self.action_timestep
					
					current_psv += psv

		return current_psv

	def get_personal_space_violation(self):
		rob_pos = np.array(self.robots[0].get_position())[:2]
		rob_vel = np.linalg.norm(self.robots[0].get_velocity()[:2])
		
		current_psv = 0
		
		# First get static PSV to the nearest pedestrian
		min_distance = np.inf
		
		for location in self.pedestrians:
			for pid in self.pedestrians[location]:
				curr_ped = self.pedestrians[location][pid]['object']
								
				dist = np.linalg.norm(rob_pos - curr_ped.get_position()[:2]) - (self.pedestrians_radius + self.robot_radius)
				
				if dist < min_distance:
					min_distance = dist
		
			if min_distance < self.personal_space:
				current_psv += (self.personal_space - min_distance) / self.personal_space
				
# 		# Now add a velocity prediction component
# 		min_future_distance = np.inf
# 		self.lookahead_interval = 3.0
# 		
# 		for location in self.pedestrians:
# 			for pid in self.pedestrians[location]:
# 				curr_ped = self.pedestrians[location][pid]['object']
# 				
# 				ped_pos = curr_ped.get_position()[:2]
# 				ped_vel = curr_ped.get_velocity()[:2]
# 		
# 				rel_pos = ped_pos - rob_pos
# 				rel_vel = ped_vel - rob_vel
# 				
# 				future_rel_pos = rel_pos + rel_vel * self.lookahead_interval
# 								
# 				dist = np.linalg.norm(future_rel_pos) - (self.pedestrians_radius + self.robot_radius)
# 				
# 				if dist < min_future_distance:
# 					min_future_distance = dist
# 		
# 			if min_future_distance < self.personal_space:
# 				current_psv += (self.personal_space - min_future_distance) / self.personal_space				
# 											
# 			if current_psv > 0:
# 				# weight by time over which action takes place
# 				current_psv *= self.action_timestep
# 																						
		return current_psv

	def get_customized_reward(self, collision_links=[], action=None, info={}):	
		if self.psv_reward_weight != 0:			
			psv = self.get_personal_space_violation()
			self.personal_space_violations += psv
			reward = self.psv_reward_weight * psv
		else:
			reward = 0
		
		return reward
	
	def customized_step(self):
		if self.use_crowdnav_scenario:
			self.customized_step_crowdnav()
		else:
			self.customized_step_svl()

	def customized_step_crowdnav(self):
		if self.use_pedestrian_behaviors:
			self.get_pedestrian_chats()
		pedestrians_actions = dict()
		min_distance_stepwise = 10 ** 5
		has_personal_space_violation = False
		rob_pos = np.array(self.robots[0].get_position())[:2]
		if self.save_trajectories:
			# if self.current_episode not in self.robot_trajectories:
				# self.robot_trajectories[self.current_episode] = []
			self.robot_trajectories.append(rob_pos)

		if self.agent_location == 'crowdnav_circle_crossing':
			location = 'crowdnav_circle_crossing'
		else:
			location = 'crowdnav_square_crossing'
		
		for pid in self.pedestrians[location]:
			curr_ped = self.pedestrians[location][pid]['object']
			ped_pos = curr_ped.get_position()

			if self.save_trajectories:
				# if self.current_episode not in self.pedestrian_trajectories:
					# self.pedestrian_trajectories[self.current_episode] = dict()
				if pid not in self.pedestrian_trajectories:
					self.pedestrian_trajectories[pid] = []
				self.pedestrian_trajectories[pid].append(ped_pos)

			# Check for personal space violation
			distance = l2_distance(rob_pos, ped_pos) - (self.pedestrians_radius + self.robot_radius)
	
# 			if distance < min_distance_stepwise:
# 				min_distance_stepwise = distance
# 			if distance < self.personal_space:
# 				has_personal_space_violation = True
# 				self.personal_space_violation_step += (self.personal_space - distance)
					
			# Schedule pedestrian's next action.
			ob = []
			for other_pid in self.pedestrians[location]:
				if other_pid != pid:
					other_ped = self.pedestrians[location][other_pid]['object']
					ob.append(other_ped.get_observable_state())
			if self.pedestrians_can_see_robot:
				ob += [self._get_robot_observable_state()]
			if self.walls:
				walls_config = list(zip(self.walls['walls_pos'], self.walls['walls_dim']))
			else:
				walls_config = list()
							
			# Stop pedestrian when goal is reached
			if l2_distance(curr_ped.get_goal_position(), ped_pos) < self.pedestrians_dist_tol:
				curr_ped_action = ActionXY(vx=0, vy=0)
											
			self.pedestrians_actions[(location, pid)] = curr_ped.act(ob, walls=walls_config, obstacles=[], allow_backward_motion=True)
		
#		self.personal_space_violation += self.personal_space_violation_step
#		 if has_personal_space_violation:
#			 self.n_personal_space_violations += 1
#		self.total_personal_space += min_distance_stepwise
#		if min_distance_stepwise < self.min_personal_space:
#			self.min_personal_space = min_distance_stepwise
			
	def customized_step_svl(self):
		if self.use_pedestrian_behaviors:
			self.get_pedestrian_chats()
		pedestrians_actions = dict()
		self.pedestrians_actions = dict()
		min_distance_stepwise = 10 ** 5
		has_personal_space_violation = False
#		self.personal_space_violation_step = 0
		rob_pos = np.array(self.robots[0].get_position())[:2]
		# if self.save_trajectories:
			# if self.current_episode not in self.robot_trajectories:
				# self.robot_trajectories[self.current_episode] = []
			# self.robot_trajectories.append(rob_pos)

		for location in self.pedestrians:
			for pid in self.pedestrians[location]:
				curr_ped = self.pedestrians[location][pid]['object']
				ped_pos = curr_ped.get_position()

				if self.keep_trajectory:
					vel_alt = curr_ped.get_velocity()
					vel_alt = float(np.linalg.norm(vel_alt))
					ped_pos_float = [float(p) for p in ped_pos]
					self.trajectory['ped_pos'][pid].append(ped_pos_float)
					self.trajectory['ped_linear_vel'][pid].append(vel_alt)
					# self.trajectory['ped_angular_vel'][pid].append(angular_vel)

				# Check for personal space violation.
				# if l2_distance(rob_pos, curr_ped.get_position()) < self.personal_space:
					# self.n_personal_space_violations += 1
				# Only consider pedestrians in the current locations
				if location == self.agent_location:
					# print('pedestrian {} location {} = robot location {}'.format(pid, location, self.agent_location))
					distance = l2_distance(rob_pos, ped_pos) - (self.pedestrians_radius + self.robot_radius)

# 					if distance < self.personal_space:
# 						has_personal_space_violation = True
# 						self.personal_space_violation_step += (self.personal_space - distance)

				# Check if pedestrian reaches the goal.
				if l2_distance(curr_ped.get_goal_position(), ped_pos) < self.pedestrians_dist_tol:
					if self.use_pedestrian_behaviors:
						if curr_ped.target_location not in self.ped_waypoints and self.global_time - curr_ped.pause_start_time > curr_ped.pause_interval:
							curr_ped.pause_start_time = self.global_time
							curr_ped.pause_interval = np.random.uniform(curr_ped.min_pause_interval, curr_ped.max_pause_interval)
							self.reset_single_pedestrian(location, pid)
					else:
						self.reset_single_pedestrian(location, pid)		
				else:
					if np.linalg.norm(curr_ped.get_velocity()) < 0.01:
						self.pedestrians[location][pid]['stop_timestamp'] += 1
						# A hardcoded heuristic threshold for resetting stopped pedestrian.
						if self.pedestrians[location][pid]['stop_timestamp'] >= self.max_step:
							print('Reset pedestrian {} because of stopping {}'.format(pid, self.pedestrians[location][pid]['stop_timestamp']))
							self.reset_single_pedestrian(location, pid)
							self.pedestrians[location][pid]['stop_timestamp'] = 0
					else:
						self.pedestrians[location][pid]['stop_timestamp'] = 0

				# Schedule pedestrian's next action.
				ob = []
				for other_pid in self.pedestrians[location]:
					if other_pid != pid:
						other_ped = self.pedestrians[location][other_pid]['object']
						ob.append(other_ped.get_observable_state())
				if self.pedestrians_can_see_robot:
					ob += [self._get_robot_observable_state()]
				if self.walls:
					walls_config = list(zip(self.walls['walls_pos'], self.walls['walls_dim']))
				else:
					walls_config = list()

				if self.use_pedestrian_behaviors:
					for ped_pair in self.pedestrian_chats.keys():
						if pid in ped_pair:
							#curr_ped_action = ActionXY(vx=np.random.uniform(-0.01, 0.01), vy=np.random.uniform(-0.01, 0.01))
							curr_ped_action = ActionXY(vx=0, vy=0)
						
				self.pedestrians_actions[(location, pid)] = curr_ped.act(ob, walls=walls_config, obstacles=[], allow_backward_motion=False)
		
# 		self.personal_space_violation += self.personal_space_violation_step
# 		self.total_personal_space += min_distance_stepwise
# 		if min_distance_stepwise < self.min_personal_space:
# 			self.min_personal_space = min_distance_stepwise

	# What if robot is moving slow but still hits a static obstacle?
	def customized_collision_filter(self, collision_links):
		robot_velocity = self.robots[0].get_velocity()
		return np.linalg.norm(robot_velocity[:2]) > 0.05

	
	def get_sensor_states(self, state, collision_links=[]):
		super().get_sensor_states(state, collision_links)
		if 'pedestrian_position' in self.sensor_inputs:
			pedestrian_positions = []
			for location in self.pedestrians:
				for pid in self.pedestrians[location]:
					pedestrian_positions.append(self.pedestrians[location][pid]['object'].get_position())
			pedestrian_positions = np.array(pedestrian_positions)
			rob_pos = np.array(self.robots[0].get_position())[:2]
			ped_rob_relative_pos = (pedestrian_positions - rob_pos).flatten()
			state['pedestrian_position'] = ped_rob_relative_pos

		if 'pedestrian_velocity' in self.sensor_inputs:
			pedestrian_velocities = []
			for location in self.pedestrians:
				for pid in self.pedestrians[location]:
					pedestrian_velocities.append(self.pedestrians[location][pid]['object'].get_velocity())
			pedestrian_velocities = np.array(pedestrian_velocities)
			# Need to double check if z-direction has the correct vel.
			rob_vel = np.array(self.robots[0].get_velocity())[:2]
			ped_rob_diff_buffer = np.append((pedestrian_velocities - rob_vel), \
				np.zeros((pedestrian_velocities.shape[0], 1)), 1)
			ped_rob_relative_vel = np.array([rotate_vector_3d(ped_rob_diff, *self.robots[0].get_rpy())\
				for ped_rob_diff in ped_rob_diff_buffer])[:, :2]
			ped_rob_relative_vel = ped_rob_relative_vel.flatten()
			state['pedestrian_velocity'] = ped_rob_relative_vel
		
		if 'pedestrian_ttc' in self.sensor_inputs:	
			ttc = self.get_ttc()
						
			state['pedestrian_ttc'] = np.array(ttc).flatten()


	def run_simulation(self, keep_trajectory=False):
		collision_links = []

		for _ in range(self.simulator_loop):
			self.simulator_step()
			collision_links.append(list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0])))

			for location_pid in self.pedestrians_actions:
				pedestrian = self.pedestrians[location_pid[0]][location_pid[1]]['object']
				pedestrian.step(self.pedestrians_actions[location_pid])

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
		
				for location in self.pedestrians:
					for pid in self.pedestrians[location]:
						curr_ped = self.pedestrians[location][pid]['object']
						ped_pos = p.getBasePositionAndOrientation(curr_ped.gibson_id)[0]
						ped_pos = [float(p) for p in ped_pos]
						# ped_pos = curr_ped.get_position()
						vel = p.getBaseVelocity(curr_ped.gibson_id)

						linear_vel = np.array(vel[0])
						angular_vel = np.array(vel[1])
						linear_vel = float(np.sqrt(linear_vel[0] ** 2 + linear_vel[1] ** 2))
						angular_vel = float(angular_vel[2])

						self.trajectory['ped_instant_linear_vel'][pid].append(linear_vel)
						self.trajectory['ped_instant_angular_vel'][pid].append(angular_vel)
						self.trajectory['ped_instant_pos'][pid].append(ped_pos)


		# TODO: See if can filter directly
		return self.filter_collision_links(collision_links)


	def write_customized_summary(self, info):
		info['personal_space_violations'] = 0 if self.n_personal_space_violations == 0 else self.n_personal_space_violations
		#info['average_personal_space'] = 0 if self.current_step == 0 else self.total_personal_space / self.current_step
		#info['min_personal_space'] = self.min_personal_space


	def write_episodic_summary(self, info):
		episode = Episode(
				env=None,
				agent=self.robots[0].model_file,
				initial_pos=self.initial_pos,
				target_pos=self.current_target_position,
				success=float(info['success']),
				collision=float(info['collision']),
				timeout=float(info['timeout']),
				personal_space_violations=info['personal_space_violations'],
				#average_personal_space=info['average_personal_space'],
				#min_personal_space=info['min_personal_space'],
				geodesic_distance=0,
				shortest_path=0,
				agent_trajectory=None,
				object_files=None,
				object_trajectory=None,
				path_efficiency=0,
				kinematic_disturbance=0,
				dynamic_disturbance_a=0,
				dynamic_disturbance_b=0,
				collision_step=0,
				# robot_trajectories=self.robot_trajectories,
				# pedestrian_trajectories=self.pedestrian_trajectories,
				robot_trajectories=None,
				pedestrian_trajectories=None,
				)
		return episode
