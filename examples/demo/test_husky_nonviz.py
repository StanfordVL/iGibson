from gibson2.envs.husky_env import HuskyNavigateEnv
from gibson2.utils.play import play
import argparse
import os
from gibson2.core.render.profiler import Profiler
import time
import numpy as np
import matplotlib.pyplot as plt

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'play', 'play_husky_nonviz.yaml')
print(config_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()
    env = HuskyNavigateEnv(config = args.config)
    env.reset()

    times = []
    for _ in range(3000):
        with Profiler("Play Env: step"):
            start = time.time()
            obs, rew, env_done, info = env.step(0)
            t = time.time() - start
            times.append(t)


    plt.plot(times)
    plt.show()