from gibson2.envs.locomotor_env import NavigateRandomEnv
from gibson2.utils.trans import quaternion_from_euler
from gibson2.core.physics.interactive_objects import YCBObject
from time import time
import numpy as np
from time import time
import gibson2
import os
from gibson2.core.render.profiler import Profiler
import logging
import mmap
import contextlib

#logging.getLogger().setLevel(logging.DEBUG) #To increase the level of logging

def process_rot(rotation):
    if rotation[0] > 0:
        rotation[0] = np.pi - rotation[0]
    else:
        rotation[0] = -np.pi - rotation[0]

    if rotation[2] > 0:
        rotation[2] = np.pi - rotation[2]
    else:
        rotation[2] = -np.pi - rotation[2]

    rotation[1] = -rotation[1]
    rotation[2] = -rotation[2]

    return np.array([rotation[2], rotation[1], rotation[0]])


def main():
    config_filename = os.path.join(os.path.dirname(gibson2.__file__),
                                   '../examples/configs/turtlebot_demo.yaml')
    nav_env = NavigateRandomEnv(config_file=config_filename, mode='gui')
    nav_env.load_action_space()

    camera_pos = [[1.2, -2.85, 1.8],
                  [0.7, -2.85, 0.97],
                  [0.5, -3.5, 1.0]]
    camera_ori = [quaternion_from_euler(0.0, -1.3, np.pi),
                  quaternion_from_euler(0.0, 0.0, np.pi),
                  quaternion_from_euler(0.6, 0.0, np.pi)]
    nav_env.set_camera(camera_pos, camera_ori)

    rob_pos = [0.8, -2.85, 1.3, 0.0, 0.0, 0.0, 1.0, 0.0]
    nav_env.set_robot(rob_pos)
    rob_pos = np.array(rob_pos)

    nav_env.reset()

    obj = YCBObject('005_tomato_soup_can')
    nav_env.simulator.import_object(obj)
    obj.set_position_orientation([1.45, -2.85, 1.5], [0, 0, 0, 1])

    while True:
        with open('/home/jeremy/Desktop/my_iGibson/test_1.dat', 'r') as f_1, open('/home/jeremy/Desktop/my_iGibson/test_2.dat', 'r') as f_2:
            with contextlib.closing(mmap.mmap(f_1.fileno(), 1024, access=mmap.ACCESS_READ)) as m_1, contextlib.closing(mmap.mmap(f_2.fileno(), 1024, access=mmap.ACCESS_READ)) as m_2:
                with Profiler('Env action step'):

                    s = m_1.read(1024)
                    s2 = m_2.read(1024)

                    s = s.decode()
                    s = s.replace('\x00', '')
                    s = s.split('/')

                    s2 = s2.decode()
                    s2 = s2.replace('\x00', '')
                    s2 = s2.split('/')

                    rotation = process_rot([float(s[3]), float(s[4]), float(s[5])])
                    rotation = quaternion_from_euler(rotation[0], rotation[1], rotation[2])

                    rotation2 = process_rot([float(s2[3]), float(s2[4]), float(s2[5])])
                    rotation2 = quaternion_from_euler(rotation2[0], rotation2[1], rotation2[2])

                    del_action = np.array([-float(s[0]), -float(s[1]), float(s[2]), 0.0, 0.0, 0.0, 0.0, 0.0]) * 2.0
                    del_action[3:-1] = rotation
                    del_action[-1] = float(s[-1])

                    del_action2 = np.array([-float(s2[0]), -float(s2[1]), float(s2[2]), 0.0, 0.0, 0.0, 0.0, 0.0]) * 2.0
                    del_action2[3:-1] = rotation2
                    del_action2[-1] = float(s2[-1])

                    state, reward, done, info = nav_env.step_old(del_action, del_action2)
                    if done:
                        logging.info("Episode finished after {} timesteps".format(i + 1))
                        break


if __name__ == "__main__":
    main()
