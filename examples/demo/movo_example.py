from gibson2.core.physics.robot_locomotors import Locobot, Turtlebot, JR2_Kinova, Fetch, Movo
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

    config = parse_config('../configs/movo_interactive_nav_s2r_mp.yaml')
    movo = Movo(config)
    movo.load()
    movo.robot_specific_reset()

    config = parse_config('../configs/fetch_p2p_nav.yaml')
    fetch = Fetch(config)
    fetch.load()
    fetch.robot_specific_reset()
    fetch.set_position([0,1,0])


    for _ in range(24000):  # keep still for 10 seconds
        movo.tuck()
        fetch.tuck()
        fetch.apply_action([0,0,0,0,0,0,0,0,0,0])
        p.stepSimulation()
        time.sleep(1./240.)

    # for _ in range(2400):  # move with small random actions for 10 seconds
    #     for robot, position in zip(robots, positions):
    #         action = np.random.uniform(-1, 1, robot.action_dim)
    #         robot.apply_action(action)
    #     p.stepSimulation()
    #     time.sleep(1./240.0)

    p.disconnect()


if __name__ == '__main__':
    main()

