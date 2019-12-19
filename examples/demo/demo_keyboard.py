from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv
from time import time
import numpy as np
from time import time
import gibson2
import os
import pygame

pygame.init()
size = width, height = 256, 256
screen = pygame.display.set_mode(size)  # window needed to capture input from keyboard


def get_pressed(unicode_keys):
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    for key in unicode_keys:
        if len(key) == 0:
            return key
        key_char = chr(key[0])
        key_const = getattr(pygame, 'K_{}'.format(key_char))
        if keys[key_const]:
            return key


if __name__ == "__main__":
    config_filename = os.path.join(os.path.dirname(gibson2.__file__),
                                   '../examples/configs/turtlebot_p2p_nav_house.yaml')
    nav_env = NavigateRandomEnv(config_file=config_filename, mode='gui')
    agent = nav_env.robots[0]  # Defined in the config file
    for j in range(2):
        nav_env.reset()
        for i in range(300):    # 300 steps, 30s world time
            s = time()
            key = get_pressed(agent.keys_to_action)  # works only for discrete action space
            action = agent.keys_to_action[key]
            ts = nav_env.step(action)
            print(ts, 1 / (time() - s))
            if ts[2]:
                print("Episode finished after {} timesteps".format(i + 1))
                break
    nav_env.clean()