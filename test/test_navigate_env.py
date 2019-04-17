from gibson2.core.physics.robot_locomotors import *
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
import gibson2
from gibson2.utils.utils import parse_config
from gibson2.envs.base_env import BaseEnv
from gibson2.envs.locomotor_env import *
from time import time
from tf_agents.environments import gym_wrapper
from tf_agents.environments import utils

def test_env():
    config_filename = os.path.join(os.path.dirname(gibson2.__file__), '../test/test.yaml')
    nav_env = NavigateEnv(config_file=config_filename, mode='headless')
    for j in range(2):
        nav_env.reset()
        for i in range(300): # 300 steps, 30s world time
            s = time()
            action = nav_env.action_space.sample()
            ts = nav_env.step(action)
            print(ts, 1/(time()-s))
            if ts[2]:
                print("Episode finished after {} timesteps".format(i + 1))
                break

def test_wrapper():
    config_filename = os.path.join(os.path.dirname(gibson2.__file__), '../test/test.yaml')
    nav_env = NavigateEnv(config_file=config_filename, mode='headless')
    tfenv = gym_wrapper.GymWrapper(
        nav_env,
        discount=1,
        spec_dtype_map=None,
        auto_reset=True,
    )
    print("action spec", tfenv.action_spec())
    print("observation spec", tfenv.observation_spec())
    utils.validate_py_environment(tfenv, episodes=2)