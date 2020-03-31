import os, sys, logging
from time import time
import numpy as np
import pdb
from flex_gym.flex_vec_env import set_flex_bin_path, FlexVecEnv
from autolab_core import YamlConfig

class Simulator_Flex:
	def __init__(self, config_path, bin_path):
		logging.getLogger().setLevel(logging.INFO)
		self.config = YamlConfig(config_path)
		set_flex_bin_path(bin_path)
		self.env = FlexVecEnv(self.config)
		self.obs = self.env.reset()
	
	def reset(self, agents_to_reset=None, get_info=False):
		self.env.reset(agents_to_reset=None, get_info=False)

	def add_object(self, name, pos, scale):
		self.env.add_object(name, pos, scale)

	def add_object_simple(self, name, pos, scale):
		self.env.add_object_simple(name, pos, scale)

	def add_object_force(self, idx, pos):
		self.env.add_object_force(idx, pos)

	def set_object_force(self, idx, pos):
		self.env.set_object_force(idx, pos)

	def set_object_point_force(self, idx, pos, force):
		self.env.set_object_point_force(idx, pos, force)

	def attach_visual_shape(self, name, idx, pos,rot, scale):
		self.env.attach_visual_shape(name, idx, pos,rot, scale)

	def attach_visual_obj(self, name, idx, pos,rot, scale):
		self.env.attach_visual_obj(name, idx, pos,rot, scale)

	def get_object_pose(self, idx):
		self.env.get_object_pose(idx)

	def get_object_contacts(self):
		self.env.get_object_contacts()

	def set_object_pose(self, idx, pose):
		self.env.set_object_pose(idx, pose)

	def get_object_poses_in_camera(self, n):
		self.env.get_object_poses_in_camera(n)

	def erase_visual_shape(self):
		self.env.rase_visual_shape()
	
	def step(self, action):
		self.obs, _, _, _ = self.env.step(action)
