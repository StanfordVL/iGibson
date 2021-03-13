from gibson2.envs.igibson_env import iGibsonEnv
from time import time
import gibson2
import os
from gibson2.render.profiler import Profiler
import logging


def main():
    config_filename = os.path.join(gibson2.example_config_path, 'humanoid_basic.yaml')
    env = iGibsonEnv(config_file=config_filename, mode='gui')
    env.reset()
    env.robots[0].base_reset([0.0, 0.0, 1.2])
    while 1:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

    env.close()


if __name__ == "__main__":
    main()
