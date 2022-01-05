import os
import time

import numpy as np
from IPython import embed

import igibson
from igibson.robots.fetch import Fetch
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config

if __name__ == "__main__":
    s = Simulator(mode="headless", use_pb_gui=True)
    scene = EmptyScene()
    s.import_scene(scene)

    config = parse_config(os.path.join(igibson.example_config_path, "behavior_onboard_sensing_fetch.yaml"))
    robot = Fetch(s, config)
    s.import_robot(robot)

    robot.reset()
    action_dim = 11
    for i in range(action_dim):
        embed()
        for _ in range(30):
            action = np.zeros(action_dim)
            action[i] = 1.0
            robot.apply_action(action)
            s.step()
            time.sleep(0.05)
        embed()
        for _ in range(30):
            action = np.zeros(action_dim)
            action[i] = -1.0
            robot.apply_action(action)
            s.step()
            time.sleep(0.05)
