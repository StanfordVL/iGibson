from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
from gibson2.core.physics.robot_locomotors import *
from gibson2.core.physics.interactive_objects import *
import yaml
import gibson2
from time import time
from gibson2.envs.locomotor_env import NavigateEnv
import cv2

def test_env():
    config_filename = os.path.join(os.path.dirname(gibson2.__file__), '../test/goggle.yaml')
    nav_env = NavigateEnv(config_file=config_filename, mode='gui')
    for j in range(2):
        nav_env.reset()
        for i in range(10): # 10 steps, 1s world time
            s = time()
            action = nav_env.action_space.sample()
            ts = nav_env.step(action)
            if 'rgb_filled' in ts[0]:
                cv2.imshow('filled', cv2.cvtColor(np.concatenate([ts[0]['rgb'],ts[0]['rgb_filled']], axis=1), cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            print(ts[1:], 1/(time()-s))
            if ts[2]:
                print("Episode finished after {} timesteps".format(i + 1))
                break

    nav_env.clean()
