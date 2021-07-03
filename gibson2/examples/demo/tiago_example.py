from gibson2.robots.tiago_single_robot import Tiago_Single
from gibson2.robots.tiago_dual_robot import Tiago_Dual
from gibson2.utils.utils import parse_config
import os
import time
import numpy as np
import pybullet as p
import pybullet_data

def main():
    p.connect(p.GUI)
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./240.)

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)


    robots = []
    config = parse_config('../configs/tiago_single_point_nav.yaml')
    tiago = Tiago_Single(config)
    robots.append(tiago)

    config = parse_config('../configs/tiago_dual_point_nav.yaml')
    tiago = Tiago_Dual(config)
    robots.append(tiago)

    positions = [
        [1, 0, 0],
        [0, 1, 0],
    ]

    for robot, position in zip(robots, positions):
        robot.load()
        robot.set_position(position)
        robot.robot_specific_reset()
        robot.keep_still()

    secs = 2
    print("Keep still for {} seconds".format(secs))
    for _ in range(240 * secs):
        p.stepSimulation()
        time.sleep(1./240.)

    secs = 30
    print("Small movements for {} seconds".format(secs))
    for _ in range(240 * secs):  # move with small random actions for 10 seconds
        for robot, position in zip(robots, positions):
            action = np.random.uniform(-1, 1, robot.action_dim)
            #action = np.zeros(robot.action_dim)
            #x = 0
            #y = robot.wheel_dim
            #action[x:y] = 0.1
            #x = y
            #y += robot.torso_lift_dim
            #action[x:y] = 0.2
            #x = y
            #y += robot.head_dim
            #action[x:y] = 0.3

            robot.apply_action(action)
        p.stepSimulation()
        time.sleep(1./240.0)

    p.disconnect()


if __name__ == '__main__':
    main()

