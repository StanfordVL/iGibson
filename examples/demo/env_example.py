from gibson2.envs.igibson_env import iGibsonEnv
from time import time
import gibson2
import os
from gibson2.render.profiler import Profiler
import logging


def main():
    config_filename = os.path.join(
        os.path.dirname(gibson2.__file__),
        '../examples/configs/turtlebot_demo.yaml')
    env = iGibsonEnv(config_file=config_filename, mode='gui')
    for j in range(10):
        env.reset()
        for i in range(100):
            with Profiler('Environment action step'):
                action = env.action_space.sample()
                state, reward, done, info = env.step(action)
                if done:
                    logging.info(
                        "Episode finished after {} timesteps".format(i + 1))
                    break
    env.close()


if __name__ == "__main__":
    main()
