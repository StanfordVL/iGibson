from gibson2.envs.locomotor_env import NavigationRandomEnv
from time import time
import numpy as np
from time import time
import gibson2
import os
from gibson2.render.profiler import Profiler
import logging

#logging.getLogger().setLevel(logging.DEBUG) #To increase the level of logging

def main():
    config_filename = os.path.join(os.path.dirname(gibson2.__file__),
                                   '../examples/configs/turtlebot_demo.yaml')
    nav_env = NavigationRandomEnv(config_file=config_filename, mode='gui')
    for j in range(10):
        nav_env.reset()
        for i in range(100):
            with Profiler('Environment action step'):
                action = nav_env.action_space.sample()
                state, reward, done, info = nav_env.step(action)
                if done:
                    logging.info("Episode finished after {} timesteps".format(i + 1))
                    break

if __name__ == "__main__":
    main()
