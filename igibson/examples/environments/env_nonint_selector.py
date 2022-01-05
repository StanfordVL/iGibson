import logging
import os
import random
from sys import platform

import yaml

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import get_available_g_scenes
from igibson.utils.utils import let_user_pick


def main(random_selection=False):
    """
    Prompts the user to select any available non-interactive scene and loads a turtlebot into it.
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    config_filename = os.path.join(igibson.example_config_path, "turtlebot_static_nav.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    # Reduce texture scale for Mac.
    if platform == "darwin":
        config_data["texture_scale"] = 0.5
    available_g_scenes = get_available_g_scenes()
    scene_id = available_g_scenes[let_user_pick(available_g_scenes, random_selection=random_selection) - 1]
    env = iGibsonEnv(config_file=config_filename, scene_id=scene_id, mode="gui_interactive")
    for j in range(10):
        logging.info("Resetting environment")
        env.reset()
        for i in range(100):
            with Profiler("Environment action step"):
                action = env.action_space.sample()
                state, reward, done, info = env.step(action)
                if done:
                    logging.info("Episode finished after {} timesteps".format(i + 1))
                    break
    env.close()


if __name__ == "__main__":
    main()
