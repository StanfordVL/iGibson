from gibson2.envs.igibson_env import iGibsonEnv
from time import time
import time as t
import gibson2
import os
import numpy as np
from gibson2.render.profiler import Profiler
import logging
import mmap
import contextlib
import math

_NEXT_AXIS = [1, 2, 0, 1]
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i+parity-1] + 1
    k = _NEXT_AXIS[i-parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    if repetition:
        q[0] = cj*(cc - ss)
        q[i] = cj*(cs + sc)
        q[j] = sj*(cc + ss)
        q[k] = sj*(cs - sc)
    else:
        q[0] = cj*cc + sj*ss
        q[i] = cj*sc - sj*cs
        q[j] = cj*ss + sj*cc
        q[k] = cj*cs - sj*sc
    if parity:
        q[j] *= -1.0

    return q


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
    config_filename = os.path.join(gibson2.example_config_path, 'humanoid_basic.yaml')
    env = iGibsonEnv(config_file=config_filename, mode='gui')
    env.reset()
    env.robots[0].base_reset([0.62, -1.05, 0.5])
    env.simulator_step()
    env.robots[0].pose_reset([0.4, -0.2, 0.3], [0.0, 0.0, 0.0, 1.0])
    env.simulator_step()

    while 1:
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

                    del_action = np.array([0.0, 0.0, 0.0, -float(s[0]), -float(s[1]), float(s[2]), 0.0, 0.0, 0.0, 0.0, 0.0]) * 3.0
                    del_action[6:-1] = rotation
                    del_action[-1] = (1.0 - float(s[-1])) * 0.9 + 0.1

                    del_action2 = np.array([0.0, 0.0, 0.0, -float(s2[0]), -float(s2[1]), float(s2[2]), 0.0, 0.0, 0.0, 0.0, 0.0]) * 3.0
                    del_action2[6:-1] = rotation2
                    del_action2[-1] = (1.0 - float(s2[-1])) * 0.9 + 0.1

                    state, reward, done, info = env.step(del_action)

    env.close()


if __name__ == "__main__":
    main()
