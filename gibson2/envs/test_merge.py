import os
import numpy as np
import gibson2
from social_locomotor_env import NavigateEnv

"""
config_path = os.path.abspath(gibson2.__file__).split('/')
config_path = '/'.join(config_path[:-2] + ['examples', 'configs', \
	'turtlebot_differential_drive_p2p_nav.yaml'])
test_env = NavigateEnv(config_path, mode='gui', action_timestep=1/10.0, physics_timestep=1/40.0)

test_env.reset()

for i in range(1000):
	action = test_env.action_space.sample()
	test_env.step(action)

test_env.clean()
"""

config_path = os.path.abspath(gibson2.__file__).split('/')
config_path = '/'.join(config_path[:-2] + ['examples', 'configs', \
	'jr2_differential_drive_p2p_nav.yaml'])

test_env = NavigateEnv(config_path, mode='gui', action_timestep=1/10.0, physics_timestep=1/40.0)

test_env.reset()

for i in range(1000):
	action = test_env.action_space.sample()
	test_env.step(action)

test_env.clean()
