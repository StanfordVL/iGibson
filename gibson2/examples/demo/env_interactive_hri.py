from gibson2.envs.igibson_env import iGibsonEnv
from time import time
import time as t
import gibson2
import os
from gibson2.render.profiler import Profiler
import logging


def main():
    config_filename = os.path.join(gibson2.example_config_path, 'humanoid_basic.yaml')
    env = iGibsonEnv(config_file=config_filename, mode='gui')
    env.reset()
    env.robots[0].base_reset([0.5, -1.0, 0.5])
    env.simulator_step()
    env.robots[0].pose_reset([0.4, -0.2, 0.3], [0.0, 0.0, 0.0, 1.0])
    env.simulator_step()

    t.sleep(10.0)

    while 1:
        for i in range(50):
            action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, float(i) / 100.]
            state, reward, done, info = env.step(action)
        for i in range(50):
            action = [0.0, 0.0, 0.0, 0.0, 0.0, -0.01, 1.0 - float(i) / 100.]
            state, reward, done, info = env.step(action)

    env.close()


if __name__ == "__main__":
    main()
