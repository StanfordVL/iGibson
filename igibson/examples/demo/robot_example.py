from igibson.robots.locobot_robot import Locobot
from igibson.robots.turtlebot_robot import Turtlebot
from igibson.robots.jr2_kinova_robot import JR2_Kinova
from igibson.robots.fetch_robot import Fetch
from igibson.utils.utils import parse_config
import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import igibson

def main():
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./240.)

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)

    robots = []
    config = parse_config(os.path.join(igibson.example_config_path, 'fetch_reaching.yaml'))
    fetch = Fetch(config)
    robots.append(fetch)

    config = parse_config(os.path.join(igibson.example_config_path,'jr_reaching.yaml'))
    jr = JR2_Kinova(config)
    robots.append(jr)

    config = parse_config(os.path.join(igibson.example_config_path, 'locobot_point_nav.yaml'))
    locobot = Locobot(config)
    robots.append(locobot)

    config = parse_config(os.path.join(igibson.example_config_path, 'turtlebot_point_nav.yaml'))
    turtlebot = Turtlebot(config)
    robots.append(turtlebot)

    positions = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ]

    for robot, position in zip(robots, positions):
        robot.load()
        robot.set_position(position)
        robot.robot_specific_reset()
        robot.keep_still()

    for _ in range(2400):  # keep still for 10 seconds
        p.stepSimulation()
        time.sleep(1./240.)

    for _ in range(2400):  # move with small random actions for 10 seconds
        for robot, position in zip(robots, positions):
            action = np.random.uniform(-1, 1, robot.action_dim)
            robot.apply_action(action)
        p.stepSimulation()
        time.sleep(1./240.0)

    p.disconnect()


if __name__ == '__main__':
    main()
