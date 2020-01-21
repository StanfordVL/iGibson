from gibson2.core.physics.robot_locomotors import Turtlebot, Husky, Ant, Humanoid, JR2, JR2_Kinova
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
import gibson2
from gibson2.utils.utils import parse_config
from gibson2.envs.base_env import BaseEnv
from gibson2.envs.locomotor_env import NavigateRandomEnv, NavigateEnv
from time import time
import numpy as np
from time import time
import os


def test_env():
    config_filename = os.path.join(gibson2.root_path, '../test/test_house.yaml')
    nav_env = NavigateEnv(config_file=config_filename, mode='headless')
    try:
        for j in range(2):
            nav_env.reset()
            for i in range(300):    # 300 steps, 30s world time
                s = time()
                action = nav_env.action_space.sample()
                ts = nav_env.step(action)
                print(ts, 1 / (time() - s))
                if ts[2]:
                    print("Episode finished after {} timesteps".format(i + 1))
                    break
    finally:
        nav_env.clean()


def test_env_reload():
    config_filename = os.path.join(gibson2.root_path, '../test/test_house.yaml')
    nav_env = NavigateEnv(config_file=config_filename, mode='headless')
    try:
        for i in range(3):
            nav_env.reload(config_filename)
            nav_env.reset()
            for i in range(300):    # 300 steps, 30s world time
                s = time()
                action = nav_env.action_space.sample()
                ts = nav_env.step(action)
                print(ts, 1 / (time() - s))
                if ts[2]:
                    print("Episode finished after {} timesteps".format(i + 1))
                    break
    finally:
        nav_env.clean()