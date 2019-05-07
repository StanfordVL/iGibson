from gibson2.envs.locomotor_env import *
from time import time
import numpy as np
from time import time

if __name__ == "__main__":
    config_filename = os.path.join(os.path.dirname(gibson2.__file__), '../examples/configs/turtlebot_p2p_nav_house.yaml')
    nav_env = NavigateRandomEnv(config_file=config_filename, mode='gui')
    for j in range(2):
        nav_env.reset()
        for i in range(300): # 300 steps, 30s world time
            s = time()
            action = nav_env.action_space.sample()
            ts = nav_env.step(action)
            print(ts, 1/(time()-s))
            if ts[2]:
                print("Episode finished after {} timesteps".format(i + 1))
                break
    nav_env.clean()